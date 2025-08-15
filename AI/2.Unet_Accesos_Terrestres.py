# -*- coding: utf-8 -*-
"""Acc-U-NET2.ipynb

"""

__autor__ = "Andres Llanos"

"""
@author: Andres Llanos
"""

! pip install tensorflow

# Conecta con google drive para leer archivos y almacenar nuevos
from google.colab import drive
drive.mount('/content/drive')
from google.colab import files

# Librerias necesarias
import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm   #barra de progreso

seed = 123
np.random.seed = seed

# Define las dimensiones de los tiles de mosaico
IMG_WIDTH = 256
IMG_HEIGHT = 256
BANDAS = 3

# Carga los cubos de datos en formato .npy
y_train = np.load('/content/drive/MyDrive/GEE_Exports/viasTrain.npy')
x_train = np.load('/content/drive/MyDrive/GEE_Exports/trainTiles.npy')
X_test = np.load('/content/drive/MyDrive/GEE_Exports/testTiles.npy')
Y_test = np.load('/content/drive/MyDrive/GEE_Exports/viasTest.npy')

# Imprime tipos de datos  
print("Data type of y_train:", y_train.dtype)
print("Data type of x_train:", x_train.dtype)
print("Data type of X_test:", X_test.dtype)
print("Data type of Y_test:", Y_test.dtype)

print(np.min(Y_test), np.max(Y_test))

print("Shape of y_train:", y_train.shape)
print("Shape of x_train:", x_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of Y_test:", Y_test.shape)

# Divide los conjuntos de datos X_test y Y_test en conjuntos de entrenamiento 80% y validación 20% 
from sklearn.model_selection import train_test_split
X_train_split, X_val, y_train_split, y_val = train_test_split(X_test, Y_test, test_size=0.2, random_state=42)

print("Shape of X_train_split:", X_train_split.shape)
print("Shape of X_val:", X_val.shape)
print("Shape of y_train_split:", y_train_split.shape)
print("Shape of y_val:", y_val.shape)

# Construye el modelo con el input de entrada
inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, BANDAS))
# Escala las imágenes de entrada al rango [0, 1] si aún no lo están.
# Suponiendo que las imágenes de entrada son uint8 con valores de hasta 255:

s = tf.keras.layers.Lambda(lambda x: x / 255.0, output_shape=(IMG_HEIGHT, IMG_WIDTH, BANDAS))(inputs)

# U-Net Contraction path 
# Se diseñó con 64 filtros de entrada, un kernel size de 3x3, función de activación ReLU (Rectified Linear Unit) que es ampliamente utilizada 
# en redes neuronales convolucionales. El kernel initializer ‘he_normal’ es un parámetro crucial que determina el punto de partida del proceso de 
# aprendizaje de la red, lo que influye en su estabilidad, velocidad de convergencia y rendimiento general.
c1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
c1 = tf.keras.layers.SpatialDropout2D(rate=0.1)(c1) #SpatialDropout2D de 10% entre cada convulucion para evitar overfitting
c1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
c1 = tf.keras.layers.SpatialDropout2D(rate=0.1)(c1) #SpatialDropout2D de 10% entre cada convulucion para evitar overfitting
c1 = tf.keras.layers.BatchNormalization()(c1) # Adiciona Batch Normalization
p1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c1) #Maxpooling

c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.SpatialDropout2D(0.1)(c2)
c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
c2 = tf.keras.layers.SpatialDropout2D(0.1)(c2)
c2 = tf.keras.layers.BatchNormalization()(c2) 
p2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c2)

c3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.SpatialDropout2D(0.1)(c3)
c3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
c3 = tf.keras.layers.SpatialDropout2D(0.1)(c3)
c3 = tf.keras.layers.BatchNormalization()(c3)
p3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c3)

c4 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.SpatialDropout2D(0.1)(c4)
c4 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
c4 = tf.keras.layers.SpatialDropout2D(0.1)(c4)
c4 = tf.keras.layers.BatchNormalization()(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

c5 = tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.SpatialDropout2D(0.1)(c5)
c5 = tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
c5 = tf.keras.layers.SpatialDropout2D(0.1)(c5)
c5 = tf.keras.layers.BatchNormalization()(c5) 

# U-Net Expansive path
u6 = tf.keras.layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.SpatialDropout2D(0.1)(c6)
c6 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
c6 = tf.keras.layers.SpatialDropout2D(0.1)(c6)
c6 = tf.keras.layers.BatchNormalization()(c6) 

u7 = tf.keras.layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.SpatialDropout2D(0.1)(c7)
c7 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
c7 = tf.keras.layers.SpatialDropout2D(0.1)(c7)
c7 = tf.keras.layers.BatchNormalization()(c7) 

u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32, (2, 2), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (2, 2), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

u8 = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.SpatialDropout2D(0.1)(c8)
c8 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
c8 = tf.keras.layers.SpatialDropout2D(0.1)(c8)
c8 = tf.keras.layers.BatchNormalization()(c8) 

u9 = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.SpatialDropout2D(0.1)(c9)
c9 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
c9 = tf.keras.layers.SpatialDropout2D(0.1)(c9)
c9 = tf.keras.layers.BatchNormalization()(c9) 

outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

# Define métricas de pérdida personalizadas
def dice_score(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.cast(tf.keras.backend.flatten(y_true), tf.float32)
    y_pred_f = tf.cast(tf.keras.backend.flatten(y_pred), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    # Binary Cross-Entropy Loss
    bce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    # Dice Loss
    dice_loss = 1 - dice_score(y_true, y_pred)
    # Combined Loss
    return bce_loss + dice_loss

model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=bce_dice_loss,  # Combinación de pérdidas
    metrics=[dice_score, 'accuracy']  # Métricas clave
)
model.summary()

#Modelcheckpoint
checkpointer = tf.keras.callbacks.ModelCheckpoint(
    'modelP.h5',
    monitor='val_dice_score',
    mode= 'max',
    verbose=1,
    save_best_only=True
    )

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_dice_score',  
    patience=15,  # Aumenta la paciencia para dar más oportunidades
    mode='max',  # Si usas Dice/IoU
    restore_best_weights=True  # Crucial para recuperar el mejor modelo
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,  # Reduce el LR a la mitad
    patience=5,  # Espera 5 épocas sin mejora
    min_lr=1e-6  # LR mínimo permitido
)

tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir='logs',
    histogram_freq=1,  # Registra histogramas de pesos/activaciones
    update_freq='epoch'  # Reduce la sobrecarga de escritura
)

callbacks = [
    checkpointer,  # ModelCheckpoint modificado
    early_stopping,  # EarlyStopping optimizado
    reduce_lr,  # ReduceLROnPlateau añadido
    tensorboard  # TensorBoard con más detalles
]

results = model.fit(
    x_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=16,
    epochs=50,
    callbacks=callbacks)

import matplotlib.pyplot as plt

# Bloque de codigo para imprimir la funcion de perdida
plt.figure(figsize=(12, 6))
plt.plot(results.history['loss'], label='Train Loss', color='blue', linewidth=2)
plt.plot(results.history['val_loss'], label='Validation Loss', color='red', linewidth=2)

# Imprime la mejor epoca
best_epoch = np.argmin(results.history['val_loss'])
plt.axvline(x=best_epoch, color='green', linestyle=':', label=f'Best Epoch ({best_epoch})')

plt.title('Training vs Validation Loss', fontsize=16)
plt.ylabel('Loss (BCE + Dice)', fontsize=12)
plt.xlabel('Epoch', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.3)
plt.ylim(0, max(max(results.history['loss']), max(results.history['val_loss'])) * 1.1)
plt.show()

# Imprime la pérdida del modelo a lo largo de los datos de entrenamiento y validación
plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
####################################################################################################################

idx = random.randint(0, len(x_train))

preds_train = model.predict(x_train[:int(x_train.shape[0]*0.9)], verbose=1) #Toma el primer 90% de los datos de entrenamiento.
preds_val = model.predict(x_train[int(x_train.shape[0]*0.9):], verbose=1)  #Esto hace predicciones sobre el 10% restante de los datos de entrenamiento.
preds_test = model.predict(X_test, verbose=1) #Esta línea realiza predicciones en el conjunto de datos de prueba independiente (X_test). Las predicciones se almacenan en preds_test.

# Aplica un umbral a las predicciones del modelo para convertirlas en máscaras binarias (0s y 1s):
# Convierten las probabilidades de salida del modelo (que están entre 0 y 1) en predicciones de clase binaria (0 o 1)
# basándose en diferentes umbrales para cada conjunto de datos.
preds_train_t = (preds_train > 0.009).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.007).astype(np.uint8)

import matplotlib.pyplot as plt

plt.hist(preds_test.flatten(), bins=50)
plt.title('Distribution of Predicted Probabilities (Before Thresholding)')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.show()

import random
import matplotlib.pyplot as plt
import numpy as np

# Imprime los resultados
# Numero de muestras a mostrar
num_samples_to_display = 1 # Puede cambiar 
for i in range(num_samples_to_display):
   
    idx = random.randint(0, len(X_test) - 1)

    print(f"Displaying sample {idx} from the test set:")

    plt.figure(figsize=(10, 5)) 

    plt.subplot(1, 2, 1) 
    plt.imshow(X_test[idx])
    plt.title(f'Original Image (Index {idx})')
    plt.axis('off')

    plt.subplot(1, 2, 2) 

    prediction_to_show = preds_test_t[idx]
    if prediction_to_show.shape[-1] == 1:
        prediction_to_show = np.squeeze(prediction_to_show)
    plt.imshow(prediction_to_show, cmap='gray') 
    plt.title('Predicted Mask')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Instala las librerias geotile y fiona para poder unir los tiles y manipular la informacion geográfica  
! pip install geotile
! pip install fiona

imgTest = ('/content/drive/MyDrive/GEE_Exports/Mosaicos/18NXH-2025.tif')

from geotile import GeoTile

test = GeoTile(imgTest)
test.meta

test.generate_tiles(save_tiles=False, stride_x=256, stride_y=256)
test.convert_nan_to_zero()
test.normalize_tiles()

test.tile_data.shape, test.tile_data.max(), test.tile_data.min(), test.tile_data.dtype

# Aplica un umbral a las predicciones del modelo para convertirlas en máscaras binarias para exportar 
threshold = 0.009

# Predicciones 
pred_test = model.predict(test.tile_data)
pred_test = (pred_test > threshold).astype(np.uint8)
print(pred_test.shape)

test.tile_data = pred_test

test.save_tiles("prediction_tiles/")

from geotile import mosaic

# Exporta los resultados. Une los 1600 tiles en uno solo archivo .tif
mosaic('prediction_tiles/', 'pred_merged18NXH.tif')

# Importa la libreria y métodos para el calculo de indicadores de desempeño
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score

# Matrices para las metricas de precisión
y_true_flat = Y_test.flatten()
y_pred_flat = preds_test_t.flatten()

# Calcula la precisión
precision = precision_score(y_true_flat, y_pred_flat)
print(f"Precision: {precision:.4f}")

# Calcula Recall
recall = recall_score(y_true_flat, y_pred_flat)
print(f"Recall: {recall:.4f}")

# Calcula F1-score
f1 = f1_score(y_true_flat, y_pred_flat)
print(f"F1-score: {f1:.4f}")

# Calcula IoU (Jaccard Score)
iou = jaccard_score(y_true_flat, y_pred_flat)
print(f"IoU (Jaccard Score): {iou:.4f}")

# Guarda el modelo entrenado para producción con nbuevos datos
model.save('my_unet_model.keras')
print("Model saved as my_unet_model.keras")
