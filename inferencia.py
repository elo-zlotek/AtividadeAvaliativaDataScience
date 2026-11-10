import pandas as pd
from pickle import load

#Modelos salvos durante o treinamento
normalizador = load(open('modelo_normalizador.model', 'rb'))
colunas_cat_treino = load(open('colunas_cat.model', 'rb'))
kmeans = load(open('modelo_kmeans.model', 'rb'))

#Recebendo dados de uma nova residência 
dados_num_nova = {
    'CRIM': [0.03237], 'ZN': [0.0], 'INDUS': [2.18], 
    'NOX': [0.458], 'RM': [6.998], 'AGE': [45.8], 
    'DIS': [6.0622], 'RAD': [3], 'TAX': [222], 
    'PTRATIO': [18.7], 'B': [394.63], 'LSTAT': [2.94], 
    'MEDV': [33.4]
}
#Dado categórico separado (sendo '0' ou '1' como texto)
dados_cat_nova = pd.Series(['0'], name='CHAS')

nova_inst_num = pd.DataFrame(dados_num_nova)

#Normalizando os dados da nova residência (Numéricos e Categóricos)
nova_inst_num_norm = normalizador.transform(nova_inst_num)
nova_inst_num_norm = pd.DataFrame(nova_inst_num_norm, columns=nova_inst_num.columns)

#Criando os dummies da nova instância
nova_inst_cat_norm = pd.get_dummies(dados_cat_nova, prefix='CHAS', dtype=int)

#Garantindo que todas as colunas dummy do treino existam, preenchendo com 0 as que faltarem
nova_inst_cat_norm = nova_inst_cat_norm.reindex(columns=colunas_cat_treino, fill_value=0)

#Recompondo a nova instância
nova_instancia_X = pd.concat([nova_inst_num_norm, nova_inst_cat_norm], axis=1)

#Inferir a qual grupo essa nova residência pertence
grupo_inferido = kmeans.predict(nova_instancia_X)[0]
print(f"Nova residência classificada no grupo: {grupo_inferido}")

#Obter e desnormalizar os dados do centroide inferido
#Vetor do centroide correspondente ao grupo inferido
centroide_normalizado = kmeans.cluster_centers_[grupo_inferido:grupo_inferido+1]

#Separando o vetor do centroide em numérico e categórico
qtd_cols_num = len(nova_inst_num.columns)
centroide_num_norm = centroide_normalizado[:, :qtd_cols_num]
centroide_cat_norm = centroide_normalizado[:, qtd_cols_num:]

#Desnormalizando dados Numéricos usando inverse_transform
centroide_num_natural = normalizador.inverse_transform(centroide_num_norm)
centroide_num_df = pd.DataFrame(centroide_num_natural, columns=nova_inst_num.columns)

#Desnormalizando dados Categóricos 
#Binarizando tudo acima de 0.5 para identificar a categoria majoritária do grupo
centroide_cat_bin = (centroide_cat_norm > 0.5).astype(int)
centroide_cat_df = pd.DataFrame(centroide_cat_bin, columns=colunas_cat_treino)
centroide_cat_natural = pd.from_dummies(centroide_cat_df, sep='_')

print("\n Perfil Desnormalizado do Centroide (Grupo {})".format(grupo_inferido))
print("\n[Variáveis Numéricas]")
print(centroide_num_df.iloc[0])

print("\n[Variável Categórica]")
print(centroide_cat_natural.iloc[0])