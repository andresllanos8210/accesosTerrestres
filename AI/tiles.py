# -*- coding: utf-8 -*-
"""tiles.ipynb

@author: Andres Llanos
"""
# Conecta con google drive para leer archivos y almacenar nuevos
from google.colab import drive
drive.mount('/content/drive')
from google.colab import files

# Instala las librerias geotile y fiona para poder generar los 
# tiles y manipular la informacion geográfica  
! pip install GeoTile
! pip install fiona

# Librerias necesarias
from tqdm import tqdm
from geotile import GeoTile
import os
import numpy as np

# Rutas de almacenamiento de mosaicos y mascaras de entrenamiento y prueba
imgTrain = ('/content/drive/MyDrive/GEE_Exports/Mosaicos/18NWH-2025.tif')
viasTrain = ('/content/drive/MyDrive/GEE_Exports/Vectores/V18NWH.shp')
imgTest = ('/content/drive/MyDrive/GEE_Exports/Mosaicos/18NXH-2025.tif')
viasTest = ('/content/drive/MyDrive/GEE_Exports/Vectores/V18NXH.shp')

train = GeoTile(imgTrain)
train.meta

# Create the output directory if it doesn't exist
output_dir = '/content/drive/MyDrive/GEE_Exports/trainTiles'
os.makedirs(output_dir, exist_ok=True)

# Toma la variable donde se almacenan los mosaicos de entrenamiento, asigna el prefijo ‘train_’
# y define el tamaño de cada mosaico en píxeles de 256x256, stride_x=256, stride_y=256 definen el desplazamiento
train.generate_tiles('/content/drive/MyDrive/GEE_Exports/trainTiles', prefix='train_', tile_x=256, tile_y=256, stride_x=256, stride_y=256)

# Crea los directorios de almacenamiento en drive si no existen aún
output_dir = '/content/drive/MyDrive/GEE_Exports/viasTrain'
os.makedirs(output_dir, exist_ok=True)

train.rasterization(viasTrain, '/content/drive/MyDrive/GEE_Exports/viasTrain/viasTrain.tif')

# Toma la variable donde se almacenan las moascaras de entrenamiento, asigna el prefijo ‘test_’
# y define el tamaño de cada mosaico en píxeles de 256x256, stride_x=256, stride_y=256 definen el desplazamiento
train_y = GeoTile('/content/drive/MyDrive/GEE_Exports/viasTrain/viasTrain.tif')
train_y.generate_tiles('/content/drive/MyDrive/GEE_Exports/viasTrain', prefix='test_', tile_x=256, tile_y=256, stride_x=256, stride_y=256)

test = GeoTile(imgTest)
test.meta

output_dir = '/content/drive/MyDrive/GEE_Exports/testTiles'
os.makedirs(output_dir, exist_ok=True)

test.generate_tiles('/content/drive/MyDrive/GEE_Exports/testTiles', prefix='test_', tile_x=256, tile_y=256, stride_x=256, stride_y=256)

# Se genera el cubo de datos con los tiles de S2 de train
# Ejecute con save_tiles=False para cargar los tiles en memoria como una matriz 'array' de numpy 
train.generate_tiles(save_tiles=False, tile_x=256, tile_y=256, stride_x=256, stride_y=256)

# Se genera el cubo de datos con los tiles de las mascaras de vias
train_y.generate_tiles(save_tiles=False, tile_x=256, tile_y=256, stride_x=256, stride_y=256)

# Genera el cubo de datos con los tiles de las mascaras de test
test.generate_tiles(save_tiles=False, tile_x=256, tile_y=256, stride_x=256, stride_y=256)

# Estadísticas generales 
train.tile_data.min(), train.tile_data.max(), #test.tile_data.mean()
test.tile_data.min(), test.tile_data.max(), #test.tile_data.mean()
train.convert_nan_to_zero()
test.convert_nan_to_zero()

# Normalizar la información, es decir, los valores de los píxeles se escalan entre 0 y 1 para facilitar el aprendizaje
train.normalize_tiles()
test.normalize_tiles()

# Estadísticas generales 
train.tile_data.min(), train.tile_data.max()
test.tile_data.min(), test.tile_data.max()
test.tile_data.shape
test.tile_data.min(), test.tile_data.max()

# Genera la matriz de datos en formato npy
train.save_numpy('/content/drive/MyDrive/GEE_Exports/trainTiles.npy')
test.save_numpy('/content/drive/MyDrive/GEE_Exports/testTiles.npy')
train_y.save_numpy('/content/drive/MyDrive/GEE_Exports/Matriz/viasTrain.npy')
test_y.save_numpy('/content/drive/MyDrive/GEE_Exports/Matriz/viasTest.npy')

# Resumen de las características de los datos: sus valores mínimo y máximo, su tipo de datos y su forma (dimensiones)
X_train = np.load('/content/drive/MyDrive/GEE_Exports/trainTiles.npy')
X_train.max(), X_train.min(), X_train.dtype, X_train.shape

y_train= np.load('/content/drive/MyDrive/GEE_Exports/viasTrain.npy')

import matplotlib.pyplot as plt

# Despliega los tiles de mosaico en RGB con su respectiva mascara de accesos terrestres
num_samples = 2  # Numero de muestras a mostrar
fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 5)) 

for i in range(num_samples):
    img_index = np.random.randint(0, X_train.shape[0]) 

    # Muestra la imagen
    axes[i, 0].imshow(X_train[img_index]) 

    axes[i, 0].set_title(f"Image {img_index}")
    axes[i, 0].axis('on') 

# Muestra la máscara
# Usa un mapa de colores que haga que la máscara sea claramente visible, por ejemplo, "gris" o "binario".
    axes[i, 1].imshow(y_train[img_index, :, :, 0], cmap='gray') 
    axes[i, 1].set_title(f'Mask {img_index}')
    axes[i, 1].axis('on')

plt.tight_layout() # Ajusta el diseño para evitar que los títulos se superpongan
plt.show()

sample_index = np.random.randint(0, y_train.shape[0])

mask_to_display = y_train[sample_index, :, :, 0]

# Muestra la máscara de accesos terrestres con el mismo indice del tile de mosaico
plt.figure(figsize=(6, 6))
plt.imshow(mask_to_display, cmap='gray') # Usa 'gray' para las mascaras
plt.title(f'Mask Sample {sample_index}')
plt.show()

# Grafica una imagen RGB de entrada de muestra con su respectiva mascara
fig, (ax1, ax2) = plt.subplots(1,2)
img = np.random.randint(0, X_train.shape[0]) # Utiliza el número de muestras
ax1.imshow(X_train[img]) 
ax2.imshow(y_train[img, :, :, 0], cmap='gray')
ax1.set_title(f"Image  {img}")
ax2.set_title(f'Mask {img}')
plt.show()

