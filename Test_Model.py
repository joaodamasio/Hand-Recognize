import cv2
import mediapipe as mp
import numpy as np
import pickle

# Lendo o os arquivos e atribuindo eles a variáveis
modelo_dict = pickle.load(open('./modelo.p', 'rb'))
modelo = modelo_dict['modelo']

# Selecionando qual camera o opencv deve abrir assim que eu iniciar o código
cap = cv2.VideoCapture(0)

# Atribuindo as funções 
mp_maos = mp.solutions.hands
mp_desenho = mp.solutions.drawing_utils
mp_estilo_desenho = mp.solutions.drawing_styles

# Atribuindo as funções 
maos = mp_maos.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Definindo as letras do albeto
labels_dict = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'J', 10:'K', 11:'L', 12:'M', 13:'N',
               14:'O', 15:'P', 16:'Q', 17:'R', 18:'S', 19:'T', 20:'U', 21:'V', 22:'W', 23:'X', 24:'Y', 25:'Z' }
while True:

    dados_aux = []
    x_ = []
    y_ = []
    ret, frame = cap.read()

    H,W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #detectando os marcadores
    resultados = maos.process(frame_rgb)

    #iterando esses marcadores para cada frame se houver uma mao
    if resultados.multi_hand_landmarks:
        for marcadores_mao in resultados.multi_hand_landmarks:
            mp_desenho.draw_landmarks(
                frame,
                marcadores_mao,
                mp_maos.HAND_CONNECTIONS,
                mp_estilo_desenho.get_default_hand_landmarks_style(),
                mp_estilo_desenho.get_default_hand_connections_style())
            
        for marcadores_mao in resultados.multi_hand_landmarks:
            for i in range(len(marcadores_mao.landmark)):
                x = marcadores_mao.landmark[i].x
                y = marcadores_mao.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(marcadores_mao.landmark)):
                x = marcadores_mao.landmark[i].x
                y = marcadores_mao.landmark[i].y
                dados_aux.append(x-min(x_))
                dados_aux.append(y-min(y_))

        # Ajustando posição do leitor das mãos de acordo com o x e y
        x1 = int(min(x_) *W) - 10
        y1 = int(min(y_) *H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        predicao = modelo.predict([np.asarray(dados_aux)])
        predicao_caratere = labels_dict[int(predicao[0])]

        # Dando forma ao formato de reconhecimento da mão
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicao_caratere, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)
       
    # Caso a tecla 25 seja apertada o codigo é encerrado fechando a camera
    cv2.imshow('frame', frame)
    cv2.waitKey(25)

# Função para fechar

cap.release()
cv2.destroyAllWindows()