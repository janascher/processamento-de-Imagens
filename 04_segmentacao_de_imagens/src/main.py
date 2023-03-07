import cv2
import os
from matplotlib import pyplot as plt
import numpy as np
print("np:", np.__version__)
print("cv2:", cv2.__version__)


def histogram_grayscale(image):
    # Converter a imagem para escala de cinza
    img_gray = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    # Exibir a imagem em escala de cinza
    plt.figure(figsize=(15, 6))
    plt.subplot(121)
    plt.imshow(img_gray, "gray")
    plt.title("Imagem em Escala de Cinza")

    # Calcular e exibir o histograma da imagem em escala de cinza
    plt.subplot(122)
    plt.hist(img_gray.ravel(), 256, [0, 256])
    plt.title("Histograma da Imagem em Escala de Cinza")

    plt.show()

def segmentation_thresh(image):
    # Copiar imagem origial
    img_original = image.copy()

    # Converter a imagem para escala de cinza
    img_gray = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    # Suavizar ou desfocar a imagem
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # Limiar adaptativa de Otsu
    thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 199, 5)

    # Exibir a imagem origial
    plt.figure(figsize=(15, 6))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
    plt.title("Imagem Original")

    # Exibir a imagem segmentada pelo método de Otsu
    plt.subplot(122)
    plt.imshow(thresh, "gray")
    plt.title("Limiarização pelo método de Otsu")

    plt.show()


# Carregar a imagem em escala de cinza
image = cv2.imread("ball.jpg")
histogram = histogram_grayscale(image)
segmentation = segmentation_thresh(image)
