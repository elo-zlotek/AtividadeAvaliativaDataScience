import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from pickle import dump

dados = pd.read_csv('HousingData (1).csv')

dados = dados.dropna()

#Separando colunas categóricas e numéricas
dados['CHAS'] = dados['CHAS'].astype(int).astype(str)

dados_num = dados.drop(columns=['CHAS'])
dados_cat = dados['CHAS']

#Normalizando a base (Dados Quantitativos)
normalizador = MinMaxScaler()
modelo_normalizador = normalizador.fit(dados_num)

dump(modelo_normalizador, open('modelo_normalizador.model', 'wb'))

#Normaliza e convertendo para DataFrame novamente
dados_num_normalizados = modelo_normalizador.transform(dados_num)
dados_num_normalizados = pd.DataFrame(dados_num_normalizados, columns=dados_num.columns)

#Normalizando a base (Dados Categóricos)
dados_cat_normalizado = pd.get_dummies(dados_cat, prefix='CHAS', dtype=int)
dados_cat_normalizado.reset_index(drop=True, inplace=True)

dump(dados_cat_normalizado.columns, open('colunas_cat.model', 'wb'))

dados_treino = pd.concat([dados_num_normalizados, dados_cat_normalizado], axis=1)

#Determinando o número ótimo de grupos (Método do Cotovelo)
distancias = []
K_range = range(1, 11)
for k in K_range:
    kmeans_teste = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_teste.fit(dados_treino)
    distancias.append(kmeans_teste.inertia_)

#Criando e salvar o gráfico do cotovelo 
plt.plot(K_range, distancias, marker='o')
plt.title('Método do Cotovelo')
plt.xlabel('Número de clusters (K)')
plt.ylabel('Inércia (Distâncias)')
plt.savefig('grafico_cotovelo.png')
print("Gráfico do método do cotovelo salvo.")

k_otimo = 4

#Treinando o modelo definitivo e salvar
kmeans = KMeans(n_clusters=k_otimo, random_state=42, n_init=10)
modelo_kmeans = kmeans.fit(dados_treino)

dump(modelo_kmeans, open('modelo_kmeans.model', 'wb'))
print(f"Treinamento feito")