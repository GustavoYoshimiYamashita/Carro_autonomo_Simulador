"""
Módulo do sistema de reconhecimento das faixas.

=====================
REQUISITO OBRIGATÓRIO
=====================

"Detecção das faixas da pista (brancas) em um fundo cinza ou de cor semelhante (ex. asfalto).
O deslocamento deve ocorrer no centro da pista e essa detecção deve ocorrer por meio de
IA/visão computacional."

=====================
---REQUISITO BÔNUS---
=====================

"Trafegar em ambiente com baixíssima luminosidade, atendendo os mesmos requisitos básicos."

=====================
--------INPUT--------
=====================

Imagem/Video: imagem das faixas laterias da "rua"

=====================
-------OUTPUT--------
=====================

Float: distancia do carro com o centro das faixas (-x.x para esquerda, 0.0 para centro e x.x para a direita)

=====================
---LÓGICA/OBJETIVO---
=====================

Pegar a distância percebida do carro com o centro das faixas e mandar para o main.
"""


import cv2
import numpy as np

"""

    INICIANDO O SISTEMA PARA O ALGORITMO HAARCASCADE NA DETECÇÃO DE PLACAS

"""

#Criando um classificador
classificador = cv2.CascadeClassifier("myhaar3.xml")

#Definindo a fonte da letra que será imprimida na tela
fonte = cv2.FONT_HERSHEY_SIMPLEX

# Mexendo no matplotlib
#%matplotlib qt5


def leitura_camera(camera):
    img = cv2.imread(f"../Carro_autonomo_Simulador/{camera}.jpg", cv2.IMREAD_COLOR)
    width = int(img.shape[1] * 2)
    height = int(img.shape[0] * 2)

    dim = (width, height)

    img1 = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    return img1


def transformacao_warp(image):
    w = image.shape[1]
    h = image.shape[0]

    # src = np.float32([[60, h], [210, h], [160, 85], [100, 85]])
    # src = np.float32([[25, h], [250, h], [160, 85], [100, 85]])
    src = np.float32([[30, h], [240, h], [175, 0], [90, 0]])
    # dst = np.float32([[0, 0], [0, h - 1], [w-1, h-1], [w-1, 0]])
    dst = np.float32([[0, h - 1], [w - 1, h - 1], [w - 1, 0], [0, 0]])

    M = cv2.getPerspectiveTransform(src, dst)
    invM = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

    return warped, invM


def transformacao_grayscale(imagem):
    return cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)


def transformacao_threshold(imagem):
    ret, imagem = cv2.threshold(imagem, 100, 225, cv2.THRESH_BINARY)

    if ret == False:
        print("Erro no Threshold!")
    else:
        return imagem


# Esse método verifica um array do pixels para encontrar a linha da rua
def detectando_linhas_metodo2(image):
    # Fazendo a transformação Warp na imagem
    warped, invM = transformacao_warp(image)
    # Tranfosrmação grayscale
    # grayscale = camera_funcoes.transformacao_grayscale(warped)
    # Transformação Threshold
    # threshold = camera_funcoes.transformacao_threshold(grayscale)

    w = warped.shape[1]
    h = warped.shape[0]

    '''
    x1, y1 = 0, 120
    x2, y2 = 50, 120

    line_thickness = 2
    cv2.line(warped, (x1, y1), (x2, y2), (0, 255, 0), thickness=line_thickness)

    '''

    rgb_list = []
    rgb_list2 = []

    for i in range(80):

        # Desenha uma linha vermelha do lado esquerdo
        cv2.circle(image, (i, 115), radius=1, color=(0, 0, 255), thickness=-1)

        # Desenha uma linha vermelha do lado direito
        direita = w - i - 1
        cv2.circle(image, (direita, 115), radius=1, color=(0, 0, 255), thickness=-1)

        colorEsquerda = image[120, i]
        blueE = int(colorEsquerda[0])
        greenE = int(colorEsquerda[1])
        redE = int(colorEsquerda[2])

        colorDireita = image[120, direita]
        blueD = int(colorDireita[0])
        greenD = int(colorDireita[1])
        redD = int(colorDireita[2])

        if blueE > 200 and greenE > 200 and redE > 200:
            rgb_list.append([blueE, greenE, redE, i])
            cv2.circle(image, (i, 115), radius=1, color=(0, 255, 0), thickness=-1)

        if blueD > 100 and greenD > 100 and redD > 100:
            rgb_list2.append([blueD, greenD, redD, direita])
            cv2.circle(image, (direita, 115), radius=1, color=(0, 255, 0), thickness=-1)

    distancia_ideal = 80

    distancia = 0
    valor = 0

    try:
        meio = int(len(rgb_list) / 2)
        meio_pixel = rgb_list[meio][3]

        distancia = (meio_pixel - 120) * -1
        valor = distancia - distancia_ideal
        # print(f"Valor da distancia {valor}")
    except:
        pass

    return image, valor

# Esse algoritmo utiliza a técnica haarcascade para detectar a placa STOP
def detectando_placa_haarcascade(image2):
    # Convertendo a imagem para a escala de cinza
    imagemCinza = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Atribuindo as classificações a variável facesDetectadas
    placas = classificador.detectMultiScale(imagemCinza, minNeighbors=9,
                                            scaleFactor=1.5,
                                            minSize=(50, 50))

    # Nas faces dectadas, desenhar um retângulo e escrever Humano
    for (x, y, l, a) in placas:
        quadrado = x, y, l, a
        #x = int(x * 0.95)
        #y = int(y * 0.95)
        #l = int(l * 1.3)
        #a = int(a * 1.3)
        cv2.rectangle(image2, (x, y), (x + l, y + a), (0, 0, 255), 2)
        cv2.putText(image2, 'Placa', (x, y + (a + 30)), fonte, 1, (0, 255, 255))
        return quadrado


"""

    INICIANDO O SISTEMA PARA A DETCÇÃO DE CÍRCULOS VERMELHOS NA CÂMERA
    posteriormente utilizado para detectar o raio da placa "STOP"

"""

import imutils
import argparse
from collections import deque

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
                help="max buffer size")
args = vars(ap.parse_args())

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
greenLower = (0, 37, 148)
greenUpper = (100, 192, 255)
pts = deque(maxlen=args["buffer"])


# Esse algoritmo detecta círculos vermelhos na imagem
def detectandoCirculo(image2):
    global radius, y, x
    blurred = cv2.GaussianBlur(image2, (11, 11), 0)
    # hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv2.inRange(blurred, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=10)
    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None
    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        raio = radius
        # only proceed if the radius meets a minimum size
        # print(f"Raio: {radius}")
        if radius > 40:
            speed = 0

    # update the points queue
    pts.appendleft(center)
    # loop over the set of tracked points
    for i in range(1, len(pts)):
        # if either of the tracked points are None, ignore
        # them
        if pts[i - 1] is None or pts[i] is None:
            continue
        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        #cv2.line(image2, pts[i - 1], pts[i], (0, 0, 255), thickness)

    return x, y, radius, center

def detectandoCorVermelha(image2):
    '''

        MÉTODO DE DETECTAR A COR VERMELHA NA IMAGEM

    '''

    try:
        x, y, radius, center = detectandoCirculo(image2)

        # print(f"X, Y: ({x}, {y}) Raio: {radius}")
        r = int(radius * 0.3)
        x = int(x)
        y = int(y)
        coordenada1 = [x - r, y - r]
        coordenada2 = [x + r, y + r]
        # print(f"Coordenada1: {coordenada1}, coordenada2: {coordenada2}")
        # cv2.rectangle(image2, (x, y), (x + l, y + a), (0, 0, 255), 2)
        # cv2.rectangle(image2, (x - r, y - r), (x + r, y + r), (0, 0, 255), 2)

        return r, x, y, radius, center
    except:
        pass

    """

        VISÃO COMPUTACIONAL DETECTANDO PLACAS STOP NA IMAGEM

    """

def detectandoPlacaStop(image2, image3, center, r):
    """

        VISÃO COMPUTACIONAL DETECTANDO PLACAS STOP NA IMAGEM

    """

    try:
        quadrado = detectando_placa_haarcascade(image3)
        xq, yq, l, a = quadrado
        # print(f"Coordenada1C: {x - r, y - r}, coordenada2: {x + r, y + r}")
        # print(f"Coordenada1Q: [{xq}, {yq}], coordenada2: [{xq+l}, {yq+a}]")

        # Desenha um círculo apenas na placa STOP e descobre o raio
        if x - r > xq and y - r > yq and x + r < xq + l and y + r < yq + a:
            if radius > 10:
                print(radius)
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(image2, (int(x), int(y)), int(radius),
                           (0, 255, 255), 2)
                cv2.circle(image2, center, 5, (0, 0, 255), -1)
                cv2.putText(image2, 'STOP', (xq, yq + (a + 30)), fonte, 1, (0, 255, 255))
            if radius > 50:
                print("PARAR")
                return True
    except:
        pass
    cv2.imshow("Frame", image2)
    # cv2.imshow("placa", image3)
