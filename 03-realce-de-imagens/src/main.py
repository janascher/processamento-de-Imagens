import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
print("np:", np.__version__)
print("cv2:", cv2.__version__)


def histograma_grayscale(imagem):
    # Ler a imagem em escala de cinza
    img_gray = np.array(cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY))

    # Exibir a imagem original
    plt.figure(figsize=(15, 6))
    plt.subplot(121)
    plt.imshow(img_gray, "gray")
    plt.title("Imagem Original")

    # Calcular e exibir o histograma da imagem original
    plt.subplot(122)
    plt.hist(img_gray.ravel(), 256, [0, 256])
    plt.title("Histograma da Imagem Original")

    # Equalizar a imagem
    # Aumenta o contraste em toda a imagem.
    img_equalizada = cv2.equalizeHist(img_gray)

    # Exibir a imagem equalizada
    plt.figure(figsize=(15, 6))
    plt.subplot(121)
    plt.imshow(img_equalizada, "gray")
    plt.title("Imagem Equalizada")

    # Calcular e exibir o histograma da imagem equalizada
    plt.subplot(122)
    plt.hist(img_equalizada.ravel(), 256, [0, 256])
    plt.title("Histograma da Imagem Equalizada")

    # Equalizar a imagem (CLAHE)
    # Possui a vantagem de limitar o contraste em regições específicas da imagem.
    # Sendo útil quando a imagem contém áreas muito escuras ou muito claras que precisam ser ajustadas separadamente.
    clahe = cv2.createCLAHE(clipLimit=12.0, tileGridSize=(8, 8))
    img_equalizada_clahe = clahe.apply(img_gray)

    # Exibir a imagem equalizada (CLAHE)
    plt.figure(figsize=(15, 6))
    plt.subplot(121)
    plt.imshow(img_equalizada_clahe, "gray")
    plt.title("Imagem Equalizada (CLAHE)")

    # Calcular e exibir o histograma da imagem equalizada (CLAHE)
    plt.subplot(122)
    plt.hist(img_equalizada_clahe.ravel(), 256, [0, 256])
    plt.title("Histograma da Imagem Equalizada (CLAHE)")

    plt.show()


# Carregar a imagem em escala de cinza
imagem = cv2.imread("alex-folguera.jpg")
caminho_imagem = os.path.join(os.getcwd(), "img", "alex-folguera.jpg")
imagem = cv2.imread(caminho_imagem)
histograma = histograma_grayscale(imagem)
