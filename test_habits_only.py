import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

print("INICIANDO PROVA DE CONCEITO: HÁBITOS PUROS")
print("==============================================")

# 1. Carregar Dados
try:
    df = pd.read_csv('Obesity.csv')
    if len(df.columns) < 2: df = pd.read_csv('Obesity.csv', sep=';')
except:
    print("❌ Erro: Obesity.csv não encontrado.")
    exit()

# 2. Remover 'Bora remover as Muletas' (Peso, Altura e IMC)
# Vamos arrancar tudo que entrega a resposta de bandeja para ver se os hábitos realmente contam algo.
cols_to_drop = ['Weight', 'Height', 'NObeyesdad', 'Obesity']
if 'Risk_Interaction' in df.columns: cols_to_drop.append('Risk_Interaction')

# Definir Target
target_col = 'NObeyesdad' if 'NObeyesdad' in df.columns else 'Obesity'
y = df[target_col]

# Definir Features (Apenas Hábitos e Perfil)
X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

print(f"🚫 Colunas Removidas: Weight, Height (e derivados)")
print(f"✅ Colunas Mantidas (Hábitos): {list(X.columns)}")

# 3. Preparar Pipeline
categorical_features = [col for col in X.columns if X[col].dtype == 'object']
numerical_features = [col for col in X.columns if X[col].dtype != 'object']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Modelo Random Forest (Mesma configuração do original)
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42))
])

# 4. Dividir e Treinar
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n⚙️ Treinando modelo APENAS com hábitos...")
model.fit(X_train, y_train)

# 5. Avaliar
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\nResultados do Teste de Honestidade:")
print("====================================")
print(f"🎯 ACURÁCIA SÓ COM HÁBITOS: {acc:.1%}")
print("====================================")

if acc > 0.95:
    print("⚠️ ALERTA: Acurácia suspeitosamente alta! Verifique se sobrou alguma coluna de peso.")
elif acc > 0.60:
    print("✅ SUCESSO: O modelo aprendeu padrões reais! Ele consegue estimar o risco baseado no estilo de vida.")
else:
    print("⚠️ AVISO: Os hábitos sozinhos não foram suficientes para predizer com clareza.")

# 6. O que ele está olhando agora?
print("\n📊 Top 5 Hábitos mais preditivos:")
try:
    feature_names_cat = model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
    all_features = np.r_[numerical_features, feature_names_cat]
    importances = model.named_steps['classifier'].feature_importances_
    
    df_imp = pd.DataFrame({'Hábito': all_features, 'Importância': importances})
    print(df_imp.sort_values('Importância', ascending=False).head(5).to_string(index=False))
except:
    pass