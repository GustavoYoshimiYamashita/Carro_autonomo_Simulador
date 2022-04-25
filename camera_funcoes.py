import cv2
import numpy as np

def leitura_camera():
    img = cv2.imread("../Carro_autonomo_Simulador/camera1.jpg", cv2.IMREAD_COLOR)
    width = int(img.shape[1] * 2)
    height = int(img.shape[0] * 2)

    dim = (width, height)

    img1 = cv2.resize(img, dim, interpolation= cv2.INTER_AREA)

    return img1

def transformacao_warp(image):
    w = image.shape[1]
    h = image.shape[0]

    src = np.float32([[60, h], [210, h], [160, 85], [100, 85]])
    # dst = np.float32([[0, 0], [0, h - 1], [w-1, h-1], [w-1, 0]])
    dst = np.float32([[0, h - 1], [w - 1, h - 1], [w - 1, 0], [0, 0]])

    M = cv2.getPerspectiveTransform(src, dst)
    invM = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
    return warped
