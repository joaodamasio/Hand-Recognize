#importando as bibliotecas
import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

#armazenando as funções dentro de variaveis para que torne o manuzeio mais eficaz
mp_maos = mp.solutions.hands
mp_desenho = mp.solutions.drawing_utils
mp_estilo_desenho = mp.solutions.drawing_styles

maos = mp_maos.Hands(static_image_mode=True, min_detection_confidence=0.3)
#definindo o diretório aonde sera armazenado os dados
DADOS_DIR = './data'

#criando os dados e labels que serao importantes para treinar os modelos futuramente
dados = []
labels = []

#iterando por cada imagem, para que capture todos os marcadores
for dir_ in os.listdir(DADOS_DIR):
    for versao_img in os.listdir(os.path.join(DADOS_DIR, dir_)):
        dados_aux = []

        x_ = []
        y_ = []
        #convertendo imagem para imagem rgb
        img = cv2.imread(os.path.join(DADOS_DIR, dir_, versao_img))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #detectando os marcadores
        resultados = maos.process(img_rgb)

        #iterando esses marcadores para cada frame se houver uma mao
        if resultados.multi_hand_landmarks:
            for marcadores_mao in resultados.multi_hand_landmarks:
                for i in range(len(marcadores_mao.landmark)):
                    x = marcadores_mao.landmark[i].x
                    y = marcadores_mao.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(marcadores_mao.landmark)):
                    x = marcadores_mao.landmark[i].x
                    y = marcadores_mao.landmark[i].y
                    dados_aux.append(x - min(x_))
                    dados_aux.append(y - min(y_))

            dados.append(dados_aux)
            labels.append(dir_)

f = open('data.pickle', 'wb')
pickle.dump({'dados': dados, 'labels': labels}, f)
f.close()