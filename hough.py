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
    
    nombre_resultado = f"resultado_{imagen_nombre}"
    resultado_path = os.path.join(resultados, nombre_resultado)
    cv2.imwrite(resultado_path, matrizAcum)