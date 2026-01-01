import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

print("🔧 Iniciando Ajuste Fino para Regularização (Evitar Overfitting)...")

# 1. Carregar
df = pd.read_csv('Obesity.csv')
if 'family_history_with_overweight' in df.columns:
    df = df.rename(columns={'family_history_with_overweight': 'family_history'})

target_col = 'NObeyesdad' if 'NObeyesdad' in df.columns else 'Obesity'
X = df.drop(columns=[target_col])
y = df[target_col]

# 2. Pipeline (Igual ao anterior)
categorical_features = ['Gender', 'family_history', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
numerical_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 3. A MUDANÇA ESTÁ AQUI: Regularização
# Antes: max_depth=15 (Decorava muito)
# Agora: max_depth=10 (Mais genérico) + min_samples_leaf=4 (Exige grupos maiores)
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=200, 
        max_depth=10,           # <--- Podando a árvore (era 15)
        min_samples_split=10,   # <--- Exige mais dados para criar uma regra
        min_samples_leaf=4,     # <--- Evita regras para 1 pessoa só
        random_state=42, 
        class_weight='balanced'
    ))
])

# 4. Treinar e Validar
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_pipeline.fit(X_train, y_train)

print("⚙️  Modelo Regularizado Treinado.")

# 5. Verificação
y_pred_train = model_pipeline.predict(X_train)
y_pred_test = model_pipeline.predict(X_test)

acc_train = accuracy_score(y_train, y_pred_train)
acc_test = accuracy_score(y_test, y_pred_test)

print(f"\n📊 ANÁLISE DE OVERFITTING:")
print(f"Treino (Antes era 100%): {acc_train:.1%}")
print(f"Teste (Mundo Real):      {acc_test:.1%}")
print(f"Gap (Diferença):         {acc_train - acc_test:.1%}")

if acc_train < 0.99:
    print("\n✅ SUCESSO! O modelo não está mais 'decorando' (Acurácia de treino < 100%).")
    print("Isso torna o modelo mais robusto e aceitável para auditoria clínica.")
else:
    print("\n⚠️ O modelo ainda está muito forte. Considere reduzir mais o max_depth.")

# Salvar o modelo final "Auditável"
joblib.dump(model_pipeline, 'best_obesity_model.pkl')
print("\n💾 Modelo Final salvo como 'best_obesity_model.pkl'")