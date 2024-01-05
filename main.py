import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import BernoulliRBM
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

#Utilizando RBM para reconstruir a imagem extraindo suas caracteristicas


#Carregando base de dados (Imagens de digitos)
base = datasets.load_digits()
previsores = np.asarray(base.data, 'float32')
classe = base.target

#Normalizando dados
normalizador = MinMaxScaler()
previsores = normalizador.fit_transform(previsores)

#Dividindo base de dados em treino e teste
previsores_treino, previsores_teste, classe_treino, classe_teste = train_test_split(previsores, classe, test_size=0.2)

#Instanciando RBM
rbm = BernoulliRBM()
rbm.n_iter = 25 # Epocas
rbm.n_components = 50 # Neuronios da camada oculta

#Instanciando classificadores
naive_rbm = GaussianNB()
naive_simples = GaussianNB()

#Treinando classificador com RBM através de Pipeline

#x = rbm.fit_transform(previsores_treino)
# rbm.n_components - Valores da camada oculta (Imagens Redimensionada(Codificada))
# rbm.components_ - Valor reconstruido através da camada oculta (Imagem Reconstruida(Decodificada))

classificador_rbm = Pipeline(steps= [('rbm', rbm), ('naive', naive_rbm)])
classificador_rbm.fit(previsores_treino, classe_treino)

#Treinando classificador simples
naive_simples.fit(previsores_treino, classe_treino)

#Visualizando imagens geradas na camada oculta
plt.figure(figsize=(20,20))
for i, comp in enumerate(rbm.components_):
    plt.subplot(10, 10, i+1)
    plt.imshow(comp.reshape((8,8)), cmap=plt.cm.gray_r)
    plt.xticks(())
    plt.yticks(())
plt.show()

#Coletando acuracia do classificador com RBM
previsoes_rbm = classificador_rbm.predict(previsores_teste)
precisao_rbm = metrics.accuracy_score(previsoes_rbm, classe_teste)

#Coletando acuracia sem da classificador Simples
previsoes_naive = naive_simples.predict(previsores_teste)
precisao_naive = metrics.accuracy_score(previsoes_naive, classe_teste)

print('Acuracia')
print('Treino com RBM:'+str(precisao_rbm))
print('Treino sem RBM:'+str(precisao_naive))
