#importando bibliotecas
import os

import cv2

#defininfo local de armazenamento dos dados
DADOS_DIR = './data'
if not os.path.exists(DADOS_DIR):
    os.makedirs(DADOS_DIR)

numero_de_classes = 27
tamanho_dados = 100

#utilizando funcoes do cv2 e do os para acessar a camera e realizar o armazenamento das capturas
cap = cv2.VideoCapture(0)
for i in range(numero_de_classes):
    if not os.path.exists(os.path.join(DADOS_DIR,str(i))):
        os.makedirs(os.path.join(DADOS_DIR,str(i)))

    print(f'Coletando dados para classe {i}')

    feito = False
    #laço para parar de capturar a coleta de imagem apertando Q
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Pronto? Pressione "Q" !:)', (100,50),cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break


    contador = 0

    #enquanto não capturar toda base de dados necessárias o laço vai continuar
    while contador < tamanho_dados:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DADOS_DIR, str(i), f'{contador}.jpg'),frame)

        contador += 1

#quando for finalizada a captura de todos os dados a camera será fechada   
cap.release()
cv2.destroyAllWindows()