import os
import random
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from canny import cannySolo
from PIL import Image, ImageDraw

imagenes = "image_2"
resultados = "results"

if not os.path.exists(resultados):
    os.makedirs(resultados)
    
archivos = os.listdir(imagenes)

imagenes_al_azar = random.sample(archivos,5)

for imagen_nombre in imagenes_al_azar:
    imagen_path = os.path.join(imagenes, imagen_nombre)
    
    
    
    cota= 300
    x1, y1 = matrizAcum.shape
    max_local = np.zeros((x1,y1))

    for i in range(x1):
        for j in range(y1):
            if matrizAcum[i,j] == maximo_valor_global:
                print(i,j)

    xlocal, ylocal = max_local.shape

    """for i in range(xlocal):
        for j in range(ylocal):
            if max_local[i,j] == 1:
                rho = i
                theta = j
                x1 = int(rho * np.cos(np.deg2rad(theta)))
                y1 = int(rho * np.sin(np.deg2rad(theta)))
                x2 = int(x1 + 100 * np.cos(np.deg2rad(theta)))  
                y2 = int(y1 + 100 * np.sin(np.deg2rad(theta)))
                dibujante.line([(x1, y1), (x2, y2)], fill=(0, 0, 0), width=2)  # Línea negra"""
                
    # imagen = np.zeros((x,y), dtype=np.uint8)
    # rho ,theta= 447-diag ,45
    # x1 = int(rho * np.cos(np.deg2rad(theta))) -x
    # y1 = int(rho * np.sin(np.deg2rad(theta))) -y
    # x2 = int(x1 + 100 * np.cos(np.deg2rad(theta)))  -x
    # y2 = int(y1 + 100 * np.sin(np.deg2rad(theta))) - y
    # print(x1,y1,x2,y2)
    nombre_resultado = f"resultado_{imagen_nombre}"
    resultado_path = os.path.join(resultados, nombre_resultado)
    cv2.imwrite(resultado_path, matrizAcum)