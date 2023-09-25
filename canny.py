from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import cv2



def cannySolo (escgris,umbrals,umbrali):
    filas, cols =(escgris.shape[0], escgris.shape[1])
    gx = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
    gy = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
    gauss = np.array([[1, 4, 7, 4, 1], [4, 16, 26, 16, 4],[7, 26, 41, 26, 7], [4, 16, 26, 16, 4], [1, 4, 7, 4, 1]], dtype=float) / 273
    newsobel = np.zeros_like(escgris)
    newgauss = np.zeros_like(escgris)

    #suavizado gaussiano
    for i in range(2, escgris.shape[0] -2):
        for j in range(2, escgris.shape[1]-2):
            suma2=0
            for k in range(-2, 3):
                for l in range(-2, 3):
                    suma2 +=escgris[i+k, j+l] * gauss[k+2, l+2]
            newgauss[i,j] =suma2


    #Sobel
    for i in range(filas - 2):
        for j in range(cols - 2):
            gxx = np.sum(np.multiply(gx, newgauss[i:i + 3, j:j + 3]))
            gyy = np.sum(np.multiply(gy, newgauss[i:i + 3, j:j + 3]))
            newsobel[i + 1, j + 1] = np.sqrt(gxx ** 2 + gyy ** 2)



    # Supresión de no máximos
    sup = np.copy(newsobel)
    for i in range(1, filas - 1):
        for j in range(1, cols - 1):
            angle = newsobel[i, j]
            if 0 <= angle < np.pi/8 or 15*np.pi/8 <= angle <= 2*np.pi:
                y1, x1 = i, j + 1
                y2, x2 = i, j - 1
            elif np.pi/8 <= angle < 3*np.pi/8:
                y1, x1 = i - 1, j + 1
                y2, x2 = i + 1, j - 1
            elif 3*np.pi/8 <= angle < 5*np.pi/8:
                y1, x1 = i - 1, j
                y2, x2 = i + 1, j
            else:
                y1, x1 = i - 1, j - 1
                y2, x2 = i + 1, j + 1

            # Comparar con los vecinos y suprimir si no es el máximo
            if newsobel[i, j] < newsobel[y1, x1] or newsobel[i, j] < newsobel[y2, x2]:
                sup[i, j] = 0
    #umbralización
    
    finalrick=np.zeros_like(sup)

    for i in range(1, sup.shape[0] - 1):
        for j in range(1, sup.shape[1] - 1):
            if sup[i, j] > umbrals:
                finalrick[i, j] = 255
            elif umbrali < sup[i, j] <= umbrals:
                vecino = sup[i-1:i+2, j-1:j+2]
                if np.any(vecino > umbrals):
                    finalrick[i, j] = 255
    return finalrick
