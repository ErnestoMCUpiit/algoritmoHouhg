import os
import random
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

carpeta_imagenes = "images_1"
carpeta_resultados = "results_components"

if not os.path.exists(results_components):
    os.makedirs(results_components)

archivos = os.listdir(carpeta_imagenes)

imagenes_al_azar = random.sample(archivos, 5)

def pitagoras(edge_image):
    y_indices, x_indices = np.where(edge_image > 0)
    
    if len(x_indices) == 0 or len(y_indices) == 0:
        return 0
    
    min_x, max_x = min(x_indices), max(x_indices)
    min_y, max_y = min(y_indices), max(y_indices)

    cateto_x = max_x - min_x
    cateto_y = max_y - min_y

    diagonal_length = math.sqrt(cateto_x*2 + cateto_y*2)

    return diagonal_length

for imagen_nombre in imagenes_al_azar:
    imagen_path = os.path.join(carpeta_imagenes, imagen_nombre)
    
    img = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE)
    
    edge = cv2.Canny(img, 100, 200)
    
    diagonal_length = pitagoras(edge)
    
    read = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE)
    threshold_value = 150
    _, image = cv2.threshold(read, threshold_value, 255, cv2.THRESH_BINARY)

    def initialize(n):
        Parent = [i for i in range(n)]
    return Parent

    def find(Parent, x):
        if Parent[x] != x:
            Parent[x] = find(Parent, Parent[x])
        return Parent[x]

    def union(Parent, x, y):
        Parent[find(Parent, y)] = find(Parent, x)
    
    rows, cols = image.shape

    Parent = initialize(rows * cols)

    label = 0

    same = False

    while not same:
        previous = image.copy()
        label = 0

        for i in range(rows):
            for j in range(cols):
                if image[i, j] >= 1:
                    neighbors = []

                    if j > 0 and image[i, j - 1] != 0:
                        neighbors.append(image[i, j - 1])

                    if i and j > 0 and image[i - 1, j - 1] != 0:
                        neighbors.append(image[i - 1, j - 1])

                    if i > 0 and image[i - 1, j] != 0:
                        neighbors.append(image[i - 1, j])

                    if i > 0 and j < image.shape[1] - 1 and image[i - 1, j + 1] != 0:
                        neighbors.append(image[i - 1, j + 1])

                    if not neighbors:
                        label += 1
                        image[i, j] = min(label, 255)
                        Parent[label] = label
                    else:
                        min_neighbor = min(neighbors)
                        image[i, j] = min(min_neighbor, 255)
                        for neighbor in neighbors:
                            if neighbor != min_neighbor:
                                union(Parent, min_neighbor, neighbor)

        if np.array_equal(image, previous):
            same = True

    label_mapping = {}

    for i in range(rows):
        for j in range(cols):
            if image[i, j] != 0:
                root_label = find(Parent, image[i, j])
                if root_label not in label_mapping:
                    label_mapping[root_label] = len(label_mapping) + 1
                image[i, j] = label_mapping[root_label]

    for i in range(rows):
        for j in range(cols):
            if image[i, j] in label_mapping:
                image[i, j] = label_mapping[image[i, j]]

    unique_labels = np.unique(image)
    num_components = len(unique_labels)

    print(f"Componentes encontrados: {num_components}")

    components = {}
    for i in range(1, label + 1):
        root = find(Parent, i)
        if root not in components:
            components[root] = []
        components[root].append(i)

    for parent, connected_components in components.items():
        print(f"Padre {parent} tiene los hijos:")
        print(connected_components)

    colors = np.random.randint(0, 256, size=(label, 3), dtype=np.uint8)

    colored_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for i in range(1, label + 1):
        colored_image[image == i] = colors[i - 1]

    plt.imshow(colored_image)
    plt.title(f"Resultado de {imagen_nombre}")
    plt.show()

    nombre_resultado = f"resultado_{imagen_nombre}"
    resultado_path = os.path.join(carpeta_resultados, nombre_resultado)
    cv2.imwrite(resultado_path, colored_image)