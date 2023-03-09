import cv2


def detect_face(image_original):

    # Carregar o classificador cascata
    face_classifier = cv2.CascadeClassifier(
        "model/haarcascade_frontalface_default.xml")

    # Copiar imagem origial
    img_copy = image_original.copy()

    # Converter a imagem para escala de cinza
    img_gray = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)

    # Armazenar objeto(s) com as coordenadas do rosto encontrado na imagem
    faces = face_classifier.detectMultiScale(
        img_gray, 1.0485258, 6)

    if len(faces) == 0:
        print("Nenhum rosto encontrado!")
    else:
        for (x, y, w, h) in faces:
            # Desenhar um retangulo nas coordenadas encontradas pelo modelo cascata na imagem original
            face_detected = cv2.rectangle(
                image_original, (x, y), (x+w, y+h), (127, 0, 255), 2)

    # Salvar a imagem modificada com o rosto encontrado na imagem
    cv2.imwrite("images/modified/imagem1-modified-face.jpg", face_detected)

    # Exibir imagem
    cv2.imshow("Rosto Detectado", face_detected)
    cv2.imshow("Imagem Original", img_copy)

    # Permitir que as imagems apareçam e fiquem esperando na tela até que o usuário digite o comando "Esc" para fechar as imagens depois que forem visualizadas com calma
    cv2.waitKey(0)

    # Comando utilizado para destruir as imagens visualizadas e não manter elas na memória
    cv2.destroyAllWindows()


# Carregar a imagem em escala de cinza
image = cv2.imread('images/original/imagem1-original-face.jpg')
detect = detect_face(image)
