
"""

    CÓDIGO DA SIMULAÇÃO DO CARRO AUTÔNOMO

"""

from visao_computacional import camera_funcoes
from controller import *
import numpy as np
import pygame
import cv2
import matplotlib.pyplot as plt
import componentes_carro
import lidar_funcoes
import csv




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
# Definindo a velocidade do trator
speed = 5

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

teste = True


"""
    
    /////////////////////////////////////////////////////////////
    >>>>>>>>>>>>>>>>>>LOOP PRINCIPAL DO PROGRAMA<<<<<<<<<<<<<<<<<
    /////////////////////////////////////////////////////////////
    
"""

while robot.step(TIME_STEP) != -1:

    # Definindo a velocidade do trator
    componentes_carro.set_speed(speed, left_front_wheel, right_front_wheel, left_rear_wheel, right_rear_wheel)

    """
        
        DESENHANDO OS DADOS DO LIDAR E DESENHANDO NO PYGAME
        
    """

    # Atualizando a tela do Pygame para preto
    surface.fill(black)
    # Coletando onúmero de pontos do Lidar
    number_points = Lidar.getNumberOfPoints(lidar)
    # Recebendo as coordenadas dos pontos do Lidar
    list_valorX, list_valorY, list_graus, imagem = lidar_funcoes.coletando_dados_lidar(lidar, number_points)
    try:
        for x in imagem:
            if x < 5.0:
                speed = 0
            else:
                speed = 5
    except:
        pass

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
    frame, distancia = camera_funcoes.detectando_linhas_metodo2(image)

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

    r, x, y, radius, center = camera_funcoes.detectandoCorVermelha(image2)

    validadorSTOP = camera_funcoes.detectandoPlacaStop(image2, image3, center, r)

    if validadorSTOP:
        speed = 0

    print("///////////////////////")
    print(f"Curvature: {distancia}")
    print(f"Angle:     {angle}")
    print("///////////////////////", end="\n")

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







