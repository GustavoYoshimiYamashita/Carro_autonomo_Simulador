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

# Iniciando o sensor Lidar
lidar = robot.getDevice("lidar")
Lidar.enable(lidar, TIME_STEP)
Lidar.enablePointCloud(lidar)

# Variáveis para o lidar
grafico = [0.0, 0]
momento = []
leitura = []
valor = []
list_valorX = []
list_valorY = []
tamanho_lists = []
parede_frente = False
parede_direita = False
parede_esquerda = False
media_direita = 0
media_esquerda = 0
media_frente = 0
# Lista para a região norte do robô
list_norte = []
# Lista para a região sul do robô
list_sul = []
# Lista para a região leste do robô
list_leste = []
# Lista para a região oeste do robô
list_oeste = []
graus = 0.0

''' Variáveis para o Pygame'''

# Inicializando cor
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
black = (0, 0, 0)
gray = (200, 200, 200)
white = (255, 255, 255)
vertical = 600
horizontal = 600
centroY = int(vertical/2)
centroX = int(horizontal/2)
rad = 0

# Iniciando tela do Pygame
surface = pygame.display.set_mode((horizontal, vertical))

''''''''''''''''''''''''''''''

# Variáveis do PID
diferenca = 0.0
#kp = 10.0
#ki = 0.0001
#kd = -50.0
kp = 7.19999999999999 #10.0
ki =  0.00000500000000#0.0001
kd = 7.19999999999999 #-50.0
proporcional = 0.0
integral = 0.0
derivativo = 0.0
PID = 0.0
ideal_value = 180
ultimaMedida = 0.0

# Adicioando o sensor compass (Bússola) ao robô
compass = robot.getDevice('compass')
compass.enable(TIME_STEP)
direction = 0
initial_value = True


while robot.step(TIME_STEP) != -1:

    componentes_carro.set_speed(0, left_front_wheel, right_front_wheel, left_rear_wheel, right_rear_wheel)

    camera.getImage()
    camera.saveImage("camera1.jpg", 100)
    image = camera_funcoes.leitura_camera()

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



