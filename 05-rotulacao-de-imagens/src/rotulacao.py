import cv2
import numpy as np
from matplotlib import pyplot as plt
print("np:", np.__version__)
print("cv2:", cv2.__version__)


def segmentation_thresh(image):
    # Copiar imagem origial
    img_original = image.copy()

    # Converter a imagem para escala de cinza
    img_gray = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    # Aplicar threshold para binarizar a imagem
    _, thresh = cv2.threshold(
        img_gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Encontrar os contornos dos objetos na imagem binária
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Rotular cada objeto com um número inteiro
    labels = np.zeros_like(thresh)
    for i, cnt in enumerate(contours):
        cv2.drawContours(labels, contours, i, (i+1), -1)

    # Desenhar os contornos dos objetos rotulados na imagem original
    plt.imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
    plt.contour(labels, colors="blue", linewidths=1)
    plt.show()


# Carregar a imagem em escala de cinza
image = cv2.imread("photo.jpg")
segmentation = segmentation_thresh(image)
