from flask import Flask, render_template, request, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
from model_utils import entrenar_modelo, predecir_emocion
import traceback

app = Flask(__name__)
app.secret_key = 'clave_segura_123'
app.config['UPLOAD_FOLDER'] = 'dataset'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

def obtener_conteo_imagenes(emocion):
    """Función auxiliar para contar imágenes en una categoría"""
    emocion_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'train', emocion)
    if not os.path.exists(emocion_dir):
        return 0
    return len([f for f in os.listdir(emocion_dir) 
               if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

# Añadir al contexto de la plantilla
app.jinja_env.globals.update(obtener_conteo_imagenes=obtener_conteo_imagenes)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/entrenar', methods=['GET', 'POST'])
def entrenar():
    if request.method == 'POST':
        action = request.form.get('action')
        
        if action == 'upload':
            # Procesar subida de imágenes
            emocion = request.form.get('emocion')
            if not emocion:
                flash('Selecciona una emoción', 'error')
                return redirect(url_for('entrenar'))
            
            if 'files' not in request.files:
                flash('No se seleccionaron archivos', 'error')
                return redirect(url_for('entrenar'))
            
            try:
                target_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'train', emocion)
                os.makedirs(target_dir, exist_ok=True)
                uploaded = 0
                
                for file in request.files.getlist('files'):
                    if file and allowed_file(file.filename):
                        filename = secure_filename(file.filename)
                        file.save(os.path.join(target_dir, filename))
                        uploaded += 1
                
                if uploaded > 0:
                    flash(f'✅ {uploaded} imágenes subidas para {emocion}', 'success')
                else:
                    flash('⚠️ No se subieron archivos válidos', 'warning')
                    
            except Exception as e:
                flash(f'Error al subir archivos: {str(e)}', 'error')
                app.logger.error(f"Error subiendo archivos: {traceback.format_exc()}")
        
        elif action == 'train':
            # Ejecutar entrenamiento
            try:
                entrenar_modelo()
                flash('Modelo entrenado exitosamente!', 'success')
            except Exception as e:
                flash(f'Error en entrenamiento: {str(e)}', 'error')
                app.logger.error(f"Error entrenando modelo: {traceback.format_exc()}")
        
        return redirect(url_for('entrenar'))
    
    # Mostrar emociones existentes
    emociones = []
    train_path = os.path.join(app.config['UPLOAD_FOLDER'], 'train')
    if os.path.exists(train_path):
        emociones = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
    
    return render_template('entrenar.html', emociones=emociones)

@app.route('/predecir', methods=['GET', 'POST'])
def predecir():
    emocion = None
    confianza = None
    imagen_url = None
    
    if request.method == 'POST':
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                temp_path = os.path.join('static', 'temp', filename)
                os.makedirs(os.path.dirname(temp_path), exist_ok=True)
                file.save(temp_path)
                
                emocion, confianza = predecir_emocion(temp_path)
                imagen_url = f"temp/{filename}"
            except Exception as e:
                flash(f'Error al predecir: {str(e)}', 'error')
                app.logger.error(f"Error prediciendo emoción: {traceback.format_exc()}")
    
    return render_template('predecir.html', emocion=emocion, confianza=confianza, imagen_url=imagen_url)

if __name__ == '__main__':
    # Crear directorios base
    os.makedirs('dataset/train', exist_ok=True)
    os.makedirs('static/temp', exist_ok=True)
    
    # Configuración mejorada
    app.run(
        debug=True,
        use_reloader=True,
        host='0.0.0.0',
        port=5000
    )