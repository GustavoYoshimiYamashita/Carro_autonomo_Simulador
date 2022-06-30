import math



def coletando_dados_lidar(lidar, number_points):

    vertical = 300
    horizontal = 300
    centroY = int(vertical / 2)
    centroX = int(horizontal / 2)

    grafico = []
    momento = []
    valor = []
    list_valorX = []
    list_valorY = []
    list_norte = []
    list_leste = []
    list_oeste = []
    list_graus = []
    graus = 0.0
    imagem = []

    list_dados = []

    for x in range(number_points):

        #Coletando dado polar
        imagem = lidar.getRangeImage()
        #print(imagem[x])
        valor.append(imagem[x])
        momento.append(x)
        grafico = [valor, momento]
        graus = graus + 2.8125
        rad = (graus * math.pi) / 180
        list_graus.append(graus)

        # Transformando em coordenada cartesiana
        # cosseno -> x
        # seno -> y
        retaX = math.cos(rad) * imagem[x]
        retaX = (retaX * 30) + centroX
        list_valorX.append(retaX)
        retaY = math.sin(rad) * imagem[x]
        retaY = (retaY * 30) + centroY
        list_valorY.append(retaY)


    return list_valorX, list_valorY, list_graus, imagem

