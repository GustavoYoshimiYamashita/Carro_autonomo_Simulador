import camera_funcoes
from controller import *
import numpy as np
import math
import pygame
import cv2
import matplotlib.pyplot as plt
import camera_funcoes
import componentes_carro
import lidar

# Mexendo no matplotlib
#%matplotlib qt5


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

camera.enable(TIME_STEP)

# Lights
left_flasher = robot.getDevice('left_flasher')
right_flasher = robot.getDevice('right_flasher')
tail_lights = robot.getDevice('tail_lights')
work_head_lights = robot.getDevice('work_head_lights')
road_head_lights = robot.getDevice('road_head_lights')


''' Variáveis para o Pygame'''

# Inicializando cor
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
black = (0, 0, 0)
gray = (200, 200, 200)
white = (255, 255, 255)
vertical = 50
horizontal = 50
centroY = int(vertical/2)
centroX = int(horizontal/2)
rad = 0

# Iniciando tela do Pygame
surface = pygame.display.set_mode((horizontal, vertical))

''''''''''''''''''''''''''''''


while robot.step(TIME_STEP) != -1:

    componentes_carro.set_speed(0, left_front_wheel, right_front_wheel, left_rear_wheel, right_rear_wheel)

    # Pegando imagem da camera do simulador
    camera.getImage()
    # Salvando um print da imagem
    camera.saveImage("camera1.jpg", 100)
    # Fazendo a leitura da imagem com o opencv
    image = camera_funcoes.leitura_camera()
    # Fazendo a transformação Warp na imagem
    warped = camera_funcoes.transformacao_warp(image)

    img_copy = np.copy(image)
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    plt.imshow(img_copy)
    #plt.show()

    cv2.imshow("camera1", warped)

    for event in pygame.event.get():
        if event.type == 256:
            pygame.quit()
            cv2.destroyAllWindows()
            exit()



