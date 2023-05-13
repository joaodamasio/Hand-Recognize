import os

import cv2

DADOS_DIR = './data'
if not os.path.exists(DADOS_DIR):
    os.makedirs(DADOS_DIR)

numero_de_classes = 27
tamanho_dados = 100

cap = cv2.VideoCapture(0)
for i in range(numero_de_classes):
    if not os.path.exists(os.path.join(DADOS_DIR,str(i))):
        os.makedirs(os.path.join(DADOS_DIR,str(i)))

    print(f'Coletando dados para classe {i}')

    feito = False
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Pronto? Pressione "Q" !:)', (100,50),cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break


    contador = 0

    while contador < tamanho_dados:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DADOS_DIR, str(i), f'{contador}.jpg'),frame)

        contador += 1

    
cap.release()
cv2.destroyAllWindows()