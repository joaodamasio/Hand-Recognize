import pickle
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


dados_dict = pickle.load(open('./data.pickle','rb'))

dados = np.array(dados_dict['dados'])
labels = np.array(dados_dict['labels'])

X_treino,X_test, y_treino, y_test = train_test_split(dados, labels, test_size=0.2, shuffle=True, stratify=labels)

modelo = RandomForestClassifier()

modelo.fit(X_treino, y_treino)

y_previsto = modelo.predict(X_test)

pontuacao = accuracy_score(y_previsto, y_test)

print(f'{pontuacao*100} estão classificados corretos')

f = open('modelo.p', 'wb')
pickle.dump({'modelo': modelo}, f)
f.close()