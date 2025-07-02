import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from collections import Counter

def crear_modelo(input_shape=(48, 48, 1), num_classes=5):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Dropout(0.3),
        
        Conv2D(64, (3,3), activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Dropout(0.3),
        
        Conv2D(128, (3,3), activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Dropout(0.4),
        
        Flatten(),
        Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def entrenar_modelo():
    train_dir = 'dataset/train'
    
    # Verificación mejorada del dataset
    if not os.path.exists(train_dir):
        raise ValueError(f"Directorio de entrenamiento no encontrado: {train_dir}")
    
    clases = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    if not clases:
        raise ValueError("No se encontraron clases en el directorio de entrenamiento")
    
    # Conteo mejorado de imágenes
    counts = Counter()
    for clase in clases:
        class_dir = os.path.join(train_dir, clase)
        counts[clase] = len([f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    print("\nDistribución de imágenes por clase:")
    for clase, count in counts.items():
        print(f"{clase}: {count} imágenes")
        if count < 50:
            print(f"¡Advertencia! La clase {clase} tiene menos de 50 imágenes")

    # DataGenerator mejorado pero con mismo nombre
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=25,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.7, 1.3],
        validation_split=0.2,
        fill_mode='nearest'
    )

    # Generadores (mismo nombre)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        color_mode='grayscale',
        batch_size=64,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        color_mode='grayscale',
        batch_size=64,
        class_mode='categorical',
        subset='validation'
    )

    # Callbacks adicionales pero mismo flujo
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True),
        ModelCheckpoint('model/best_model.h5', monitor='val_accuracy', save_best_only=True)
    ]

    # Entrenamiento (misma estructura)
    model = crear_modelo(num_classes=len(train_generator.class_indices))
    
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=val_generator,
        validation_steps=val_generator.samples // val_generator.batch_size,
        epochs=100,
        callbacks=callbacks,
        verbose=1
    )
    
    # Guardado (mismo formato)
    os.makedirs('model', exist_ok=True)
    model.save('model/emociones.h5')
    np.save('model/class_indices.npy', train_generator.class_indices)
    
    return history

def predecir_emocion(image_path):
    if not os.path.exists('model/emociones.h5'):
        return "Modelo no encontrado", 0.0
    
    # Cargar modelo (mismo proceso)
    model = load_model('model/emociones.h5')
    
    try:
        # Cargar class_indices (mismo proceso)
        class_indices_path = 'model/class_indices.npy'
        if os.path.exists(class_indices_path):
            class_indices = np.load(class_indices_path, allow_pickle=True).item()
            emociones = list(class_indices.keys())
        else:
            emociones = ['feliz', 'triste', 'enojado', 'sorpresa', 'neutral']
            print("Advertencia: Usando clases por defecto")

        # Preprocesamiento mejorado pero misma salida
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("No se pudo leer la imagen")
            
        # Mejoras en preprocesamiento
        img = cv2.equalizeHist(img)  # Ecualización de histograma
        img = cv2.medianBlur(img, 3)  # Reducción de ruido
        img = cv2.resize(img, (48,48))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=(0, -1))
        
        # Predicción con suavizado
        pred = model.predict(img, verbose=0)[0]
        pred = np.exp(np.log(pred + 1e-7) / 0.7)  # Temperatura 0.7
        pred = pred / np.sum(pred)
        
        idx = np.argmax(pred)
        confianza = float(pred[idx])
        
        # Filtro de confianza mínima
        if confianza < 0.5:
            return "Indeterminado", confianza
            
        return emociones[idx], confianza
        
    except Exception as e:
        print(f"Error en predicción: {str(e)}")
        return "Error al procesar imagen", 0.0 