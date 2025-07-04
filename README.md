# 🎭 Sistema de Reconocimiento de Emociones

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/Flask-3.0.2-green" alt="Flask">
  <img src="https://img.shields.io/badge/TensorFlow-2.16.1-orange" alt="TensorFlow">
</p>

## 📌 Descripción
Aplicación web que clasifica emociones faciales usando una red neuronal convolucional (CNN) entrenada con TensorFlow/Keras y una interfaz Flask.

## 🚀 Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/AdierECO/proyecto-emociones.git
cd proyecto-emociones

# 2. Crear entorno virtual (Windows)
python -m venv venv
venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt
```
⚙️ Configuración
```bash
Prepara tu dataset:

Crea carpetas para cada emoción en dataset/train/:
dataset/
└── train/
    ├── feliz/
    ├── triste/
    ├── enojado/
    ├── sorpresa/
    └── neutral/
  Coloca mínimo 50 imágenes por categoría (formato JPG/PNG)

Entrenamiento del modelo:
  python app.py --action train
```
🖥️ Uso
```bash
# Iniciar la aplicación Flask
python app.py

Accede a la interfaz web en: http://localhost:5000

Funcionalidades:
Entrenar modelo: Sube imágenes y entrena la red neuronal

Predecir emociones: Sube una foto para analizar
```

📂 Estructura del Proyecto
```bash

proyecto-emociones/
├── app.py                # Aplicación principal Flask
├── model_utils.py        # Lógica de la red neuronal
├── static/               # Archivos CSS/JS
├── templates/            # Vistas HTML
├── dataset/              # Carpeta para imágenes (no incluida en repo)
└── model/                # Modelos entrenados (generados automáticamente)
```
⚠️ Notas Importantes
```bash
El dataset no está incluido en el repositorio (ver .gitignore)

Los modelos entrenados se guardan en model/ (ignorados por Git)

Requerimientos mínimos: 4GB RAM, GPU recomendada para entrenamiento
```
📄 Licencia
```bash
MIT License - Libre para uso académico y comercial.
```

<div align="center"> <p>✉️ <strong>Contacto</strong>: adierortix@gmail.com | 🌐 <a href="https://github.com/AdierECO">GitHub</a></p> </div> 
