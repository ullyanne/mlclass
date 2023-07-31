#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Atividade para trabalhar o pré-processamento dos dados.

Criação de modelo preditivo para diabetes e envio para verificação de peformance
no servidor.

@author: Aydano Machado <aydano.machado@gmail.com>
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import requests
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize

# def format_glucose(db):
#         db.loc[(db['Glucose'] <= 140), 'Glucose'] = 0
#         db.loc[(db['Glucose'] > 140) & (db['Glucose'] <= 155), 'Glucose'] = 1
#         db.loc[(db['Glucose'] > 155), 'Glucose'] = 2

def format_age(db):
        db.loc[(db['Age'] < 20), 'Age'] = 0
        db.loc[(db['Age'] >= 20) & (db['Age'] < 40), 'Age'] = 1
        db.loc[(db['Age'] >= 40) & (db['Age'] < 60), 'Age'] = 2
        db.loc[(db['Age'] >= 60), 'Age'] = 3

def format_blood_pressure(db):
        db.loc[(db['BloodPressure'] < 80), 'BloodPressure'] = 0
        db.loc[(db['BloodPressure'] >= 80) & (db['BloodPressure'] < 89), 'BloodPressure'] = 1
        db.loc[(db['BloodPressure'] >= 90) & (db['BloodPressure'] < 99), 'BloodPressure'] = 2
        db.loc[(db['BloodPressure'] > 100), 'BloodPressure'] = 3

# def format_pregnancies(db):
#         db.loc[(db['Pregnancies'] == 0), 'Pregnancies'] = 0
#         db.loc[(db['Pregnancies'] > 0) & (db['Pregnancies'] < 3), 'Pregnancies'] = 1
#         db.loc[(db['Pregnancies'] >= 3), 'Pregnancies'] = 2

# def format_bmi(db):
#         db.loc[(db['BMI'] < 24.9), 'BMI'] = 0
#         db.loc[(db['BMI'] >= 24.9) & (db['BloodPressure'] < 29.9), 'BMI'] = 1
#         db.loc[(db['BMI'] >= 30 & (db['BloodPressure'] < 34.9)), 'BMI'] = 2
#         db.loc[(db['BMI'] >= 35 & (db['BloodPressure'] < 39.9)), 'BMI'] = 3
#         db.loc[(db['BMI'] >= 40), 'BMI'] = 4

def normalize_2(db):
        db['Pregnancies'] = db['Pregnancies']/20
        db['Glucose'] = (db['Glucose']/30)
        db['BloodPressure'] = db['BloodPressure']/100
        db['SkinThickness'] = db['SkinThickness']/100
        db['Insulin'] = db['Insulin']/30
        db['BMI'] = db['BMI']/100
        db['Age'] = db['Age']/10

def fill_na(db):
        for col in db.columns:
                mean = db[col].mean()
                # Preenche os valores nulos da coluna com a média calculada
                db[col].fillna(mean, inplace=True)


print(' - Criando X e y para o algoritmo de aprendizagem a partir do arquivo diabetes_dataset')
# Caso queira modificar as colunas consideradas basta algera o array a seguir.

feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
#feature_cols = ['Glucose','Insulin', 'BMI']

print('\n - Lendo o arquivo com o dataset sobre diabetes')
data = pd.read_csv('diabetes_dataset.csv')


data_app = pd.read_csv('diabetes_app.csv')
data_app = data_app[feature_cols]


fill_na(data)
# dataset_array = data.values
# data_normalized = normalize(dataset_array)

# normalized_df = pd.DataFrame(data_normalized, columns=data.columns)
#print(normalized_df)

# # format_glucose(data)
# # format_glucose(data_app)


# format_blood_pressure(data)
# format_blood_pressure(data_app)

# format_age(data)
# format_age(data_app)

normalize_2(data)
normalize_2(data_app)

# fill_na(data)

# #format_bmi(data)
# #format_bmi(data_app)


# #format_pregnancies(data)
# #format_pregnancies(data_app)




# Criando X and y par ao algorítmo de aprendizagem de máquina.\
X = data[feature_cols]
y = data.Outcome

# #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)


# Ciando o modelo preditivo para a base trabalhada
print(' - Criando modelo preditivo')
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)

# print(' - Aplicando modelo e enviando para o servidor')
y_pred = neigh.predict(data_app)

print(' - Criando modelo preditivo')
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)

print(' - Aplicando modelo e enviando para o servidor')
y_pred = neigh.predict(data_app)

# Enviando previsões realizadas com o modelo para o servidor
URL = "https://aydanomachado.com/mlclass/01_Preprocessing.php"

#TODO Substituir pela sua chave aqui
DEV_KEY = "AiDANO"

# json para ser enviado para o servidor
data = {'dev_key':DEV_KEY,
        'predictions':pd.Series(y_pred).to_json(orient='values')}

# # accuracy = accuracy_score(y_test, y_pred)

# # print(accuracy)
# # Enviando requisição e salvando o objeto resposta
r = requests.post(url = URL, data = data)

# # Extraindo e imprimindo o texto da resposta
pastebin_url = r.text
print(" - Resposta do servidor:\n", r.text, "\n")