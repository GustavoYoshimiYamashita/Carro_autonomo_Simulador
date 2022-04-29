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
import csv

# Mexendo no matplotlib
#%matplotlib qt5

# Criando um arquivo csv para machine learning
header = ['curvature', 'center', 'angle']
data = []


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

# Keyboard
robot.keyboard.enable(TIME_STEP)


''' Variáveis para o Pygame'''

# Inicializando cor
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
black = (0, 0, 0)
gray = (200, 200, 200)
white = (255, 255, 255)
vertical = 200
horizontal = 200
centroY = int(vertical/2)
centroX = int(horizontal/2)
rad = 0

angle = 0

# Iniciando tela do Pygame
surface = pygame.display.set_mode((horizontal, vertical))

''''''''''''''''''''''''''''''

#  Função Map do arduino, regra de três
def _map(x, in_min, in_max, out_min, out_max):
    return float((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)

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

    for i in range(150):

        cv2.circle(warped, (i, 120), radius=0, color=(0, 0, 255), thickness=-1)

        color = image[120, i]
        blue = int(color[0])
        green = int(color[1])
        red = int(color[2])

        if blue > 240 and green > 240 and red > 240:
            rgb_list.append([red, green, blue, i])

    distancia = 100

    try:
        meio = int(len(rgb_list)/2)
        meio_pixel = rgb_list[meio][3]

        distancia = (meio_pixel - 150) * -1
        print(f"Valor da distancia {distancia}")
    except:
        pass

    return warped, distancia

while robot.step(TIME_STEP) != -1:

    componentes_carro.set_speed(1, left_front_wheel, right_front_wheel, left_rear_wheel, right_rear_wheel)

    # Pegando imagem da camera do simulador
    camera.getImage()
    # Salvando um print da imagem
    camera.saveImage("camera1.jpg", 100)
    # Fazendo a leitura da imagem com o opencv
    image = camera_funcoes.leitura_camera()

    frame, distancia = detectando_linhas_metodo2()

    cv2.imshow("camera1", frame)

    angle = _map(distancia, 105, 0, 0, 0.6)

    componentes_carro.set_steering_angle(angle, left_steer, right_steer)

    '''
    print("///////////////////////")
    print(f"Curvature: {curvature}")
    print(f"Center:    {center}")
    print(f"Angle:     {angle}")
    print("///////////////////////", end="\n")

    data.append([curvature, center, angle])

    componentes_carro.set_steering_angle(angle, left_steer, right_steer)
    '''

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
            with open('data.csv', 'w', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(data)
            exit()



