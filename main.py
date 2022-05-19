
"""

    CÓDIGO DA SIMULAÇÃO DO CARRO AUTÔNOMO

"""

import camera_funcoes
from controller import *
import numpy as np
import math
import pygame
import cv2
import matplotlib.pyplot as plt
import camera_funcoes
import componentes_carro
import lidar_funcoes
import csv
import imutils
import time
import argparse
from collections import deque
from imutils.video import VideoStream

"""
    
    INICIANDO O SISTEMA PARA A DETCÇÃO DE CÍRCULOS VERMELHOS NA CÂMERA
    posteriormente utilizado para detectar o raio da placa "STOP"

"""

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


"""

    INICIANDO O SISTEMA PARA O ALGORITMO HAARCASCADE NA DETECÇÃO DE PLACAS

"""

#Criando um classificador
classificador = cv2.CascadeClassifier("myhaar3.xml")

#Definindo a fonte da letra que será imprimida na tela
fonte = cv2.FONT_HERSHEY_SIMPLEX

# Mexendo no matplotlib
#%matplotlib qt5

"""
    
    INICIANDO SISTEMA PARA O MACHINE LEARNING

"""

# Criando um arquivo csv para machine learning
header = ['curvature', 'center']
data = []

"""

    INICIANDO A COMUNICAÇÃO DO WEBOTS COM O PYTHON

"""

# Definindo o tempo de atualização da tela
TIME_STEP = 10

# create the robot instance
robot = Robot()

# Velocidade máxima do robô
MAX_SPEED =  15.0 #12.3

# Devices
left_front_wheel = robot.getDevice('left_front_wheel')
right_front_wheel = robot.getDevice('right_front_wheel')
left_steer = robot.getDevice('left_steer')
right_steer = robot.getDevice('right_steer')
left_rear_wheel = robot.getDevice('left_rear_wheel')
right_rear_wheel = robot.getDevice('right_rear_wheel')

left_front_wheel.setPosition(float('inf'))
right_front_wheel.setPosition(float('inf'))
left_rear_wheel.setPosition(float('inf'))
right_rear_wheel.setPosition(float('inf'))

# Camera
camera = robot.getDevice('camera')
camera2 = robot.getDevice('camera2')

camera.enable(TIME_STEP)
camera2.enable(TIME_STEP)

# Lights
left_flasher = robot.getDevice('left_flasher')
right_flasher = robot.getDevice('right_flasher')
tail_lights = robot.getDevice('tail_lights')
work_head_lights = robot.getDevice('work_head_lights')
road_head_lights = robot.getDevice('road_head_lights')

# Keyboard
robot.keyboard.enable(TIME_STEP)

# Iniciando o sensor Lidar
lidar = robot.getDevice("lidar")
Lidar.enable(lidar, TIME_STEP)
Lidar.enablePointCloud(lidar)


"""

    INICIANDO O SISTEMA PARA O PYGAME

"""

''' Variáveis para o Pygame'''

# Inicializando cor
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
black = (0, 0, 0)
gray = (200, 200, 200)
white = (255, 255, 255)
vertical = 300
horizontal = 300
centroY = int(vertical/2)
centroX = int(horizontal/2)
rad = 0
angle = 0

raio = 0


# Iniciando tela do Pygame
surface = pygame.display.set_mode((horizontal, vertical))

''''''''''''''''''''''''''''''


"""

    FUNÇÕES E ALGORITMOS

"""

#  Função Map do arduino, regra de três
def _map(x, in_min, in_max, out_min, out_max):
    return float((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)


# Esse método desenha a área da rua
def detectando_linhas_metodo1():
    # Fazendo a transformação Warp na imagem
    warped, invM = camera_funcoes.transformacao_warp(image)
    # Tranfosrmação grayscale
    grayscale = camera_funcoes.transformacao_grayscale(warped)
    # Transformação Threshold
    threshold = camera_funcoes.transformacao_threshold(grayscale)
    # Detecção da linha
    frame, left_curverad, right_curverad = camera_funcoes.search_around_poly(threshold)
    frame = cv2.warpPerspective(frame, invM, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR)
    frame = cv2.addWeighted(frame, 0.3, image, 0.7, 0)

    # Add curvature and distance from the center
    curvature = (left_curverad + right_curverad) / 2
    # if curvature > 1500: curvature = 1500
    car_pos = image.shape[1] / 2
    # Centro da faixa 0.397m
    center = (abs(car_pos - curvature) * (3.7 / 650)) / 10
    curvatureAviso = 'Radius of Curvature: ' + str(round(curvature, 2)) + 'm'
    centerAviso = str(round(center, 3)) + 'm away from center'
    frame = cv2.putText(frame, curvatureAviso, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    frame = cv2.putText(frame, centerAviso, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    return curvature, center, frame

# Esse método verifica um array do pixels para encontrar a linha da rua
def detectando_linhas_metodo2():
    # Fazendo a transformação Warp na imagem
    warped, invM = camera_funcoes.transformacao_warp(image)
    # Tranfosrmação grayscale
    #grayscale = camera_funcoes.transformacao_grayscale(warped)
    # Transformação Threshold
    #threshold = camera_funcoes.transformacao_threshold(grayscale)

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
        meio = int(len(rgb_list)/2)
        meio_pixel = rgb_list[meio][3]

        distancia = (meio_pixel - 120) * -1
        valor = distancia - distancia_ideal
        #print(f"Valor da distancia {valor}")
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

teste = True
speed = 10

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

"""
    
    /////////////////////////////////////////////////////////////
    >>>>>>>>>>>>>>>>>>LOOP PRINCIPAL DO PROGRAMA<<<<<<<<<<<<<<<<<
    /////////////////////////////////////////////////////////////
    
"""

while robot.step(TIME_STEP) != -1:

    # Definindo a velocidade do trator
    componentes_carro.set_speed(speed, left_front_wheel, right_front_wheel, left_rear_wheel, right_rear_wheel)

    """
        
        COLETANDO OS DADOS DO LIDAR E DESENHANDO NO PYGAME
        
    """

    # Atualizando a tela do Pygame para preto
    surface.fill(black)
    # Coletando onúmero de pontos do Lidar
    number_points = Lidar.getNumberOfPoints(lidar)
    # Recebendo as coordenadas dos pontos do Lidar
    list_valorX, list_valorY, list_graus = lidar_funcoes.coletando_dados_lidar(lidar, number_points)

    # Desenhando os pontos do Lidar no Pygame
    for x in range(len(list_valorX)):
        #print(f"0,x: {dados_lidar[0][x]}, 1,x: {dados_lidar[1][x]}")
        pygame.draw.circle(surface, white, (list_valorX[x], list_valorY[x]), 5)

    # Atualizando a tela do Pygame
    pygame.display.update()
    pygame.display.flip()


    """
    
        VISÃO COMPUTACIONAL DO TRATOR
    
    """

    # Pegando imagem da camera do simulador
    camera.getImage()
    camera2.getImage()

    # Salvando um print da imagem
    # Esse método salva em tempo real uma foto do que o trator está vendo
    # apartir daí ele faz as leituras dessas imagens para aplicar a visão computacional
    camera.saveImage("camera1.jpg", 100)
    camera2.saveImage("camera2.jpg", 100)
    camera2.saveImage("camera3.jpg", 100)

    # Fazendo a leitura da imagem com o opencv
    image = camera_funcoes.leitura_camera("camera1")
    image2 = camera_funcoes.leitura_camera("camera2")
    image3 = camera_funcoes.leitura_camera("camera3")

    '''
    
        DETECTANDO AS LINHAS BRANCAS NA RUA
    
    '''

    # Utilizando o método de leitura de pixels da cor branca
    frame, distancia = detectando_linhas_metodo2()

    # Aplicando a correção no ângulo do robô
    if distancia > 0:
        angle = _map(distancia, 0, 14, 0, -0.6)
    elif distancia < 0:
        angle = _map(distancia, 0, -30, 0, 0.6)
    else:
        angle = 0

    # Enviando o ângulo para o trator
    componentes_carro.set_steering_angle(angle, left_steer, right_steer)

    # Mostrando a imagem com o método de detector de linha
    cv2.imshow("detector de linha", frame)


    '''
    
        MÉTODO DE DETECTAR A COR VERMELHA NA IMAGEM
    
    '''

    try:
        x, y, radius, center = detectandoCirculo(image2)

        #print(f"X, Y: ({x}, {y}) Raio: {radius}")
        r = int(radius * 0.3)
        x = int(x)
        y = int(y)
        coordenada1 = [x - r, y - r]
        coordenada2 = [x + r, y + r]
        #print(f"Coordenada1: {coordenada1}, coordenada2: {coordenada2}")
        #cv2.rectangle(image2, (x, y), (x + l, y + a), (0, 0, 255), 2)
        #cv2.rectangle(image2, (x - r, y - r), (x + r, y + r), (0, 0, 255), 2)

    except:
        pass


    """
    
        VISÃO COMPUTACIONAL DETECTANDO PLACAS STOP NA IMAGEM
    
    """

    try:
        quadrado = detectando_placa_haarcascade(image3)
        xq, yq, l, a = quadrado
        #print(f"Coordenada1C: {x - r, y - r}, coordenada2: {x + r, y + r}")
        #print(f"Coordenada1Q: [{xq}, {yq}], coordenada2: [{xq+l}, {yq+a}]")


        # Desenha um círculo apenas na placa STOP e descobre o raio
        if x-r > xq and y-r > yq and x+r < xq+l and y+r < yq+a:
            if radius > 10:
                print(radius)
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(image2, (int(x), int(y)), int(radius),
                           (0, 255, 255), 2)
                cv2.circle(image2, center, 5, (0, 0, 255), -1)
                cv2.putText(image2, 'STOP', (xq, yq + (a + 30)), fonte, 1, (0, 255, 255))
            if radius > 35:
                print("PARAR")
                speed = 0
    except:
        pass
    cv2.imshow("Frame", image2)
    #cv2.imshow("placa", image3)



    #print("///////////////////////")
    #print(f"Curvature: {distancia}")
    #print(f"Angle:     {angle}")
    #print("///////////////////////", end="\n")

    data.append([distancia, angle])

    componentes_carro.set_steering_angle(angle, left_steer, right_steer)

    #Esse código plota a camera com valores do eixo X e Y
    
    img_copy = np.copy(frame)
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    plt.imshow(img_copy)
    #plt.show()

    for event in pygame.event.get():
        if event.type == 256:
            pygame.quit()
            cv2.destroyAllWindows()
            for i in range(len(data)):
                print(data[i])
            with open('../ConversaoCelsiusFhrenheitRedeNeural/data.csv', 'w', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(data)
            exit()


