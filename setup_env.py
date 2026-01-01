import pandas as pd
import numpy as np
import joblib
import requests
import io
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Baixar o dataset diretamente do link raw do GitHub (mais confiável)
url = "https://raw.githubusercontent.com/pymche/Machine-Learning-Obesity-Classification/master/ObesityDataSet_raw_and_data_sinthetic.csv"
response = requests.get(url)
df = pd.read_csv(io.StringIO(response.text))
df.to_csv('Obesity.csv', index=False)
print("Dataset salvo como Obesity.csv")

# 2. Preparar o modelo (Pipeline similar ao esperado pelo código original)
target = 'NObeyesdad'
X = df.drop(columns=[target])
y = df[target]

# Identificar colunas
num_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
cat_features = [col for col in X.columns if col not in num_features]

# Preprocessamento
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])

# Pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Treino rápido
model.fit(X, y)

# Salvar modelo
joblib.dump(model, 'best_obesity_model.pkl')
print("Modelo salvo como best_obesity_model.pkl")
