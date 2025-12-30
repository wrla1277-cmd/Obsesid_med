import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

print("⏳ Sincronizando modelo com a Lógica V3...")

# 1. Carregar Dados
df = pd.read_csv('Obesity.csv')
if len(df.columns) < 2: df = pd.read_csv('Obesity.csv', sep=';')

# 2. Preparar Target e Nomes (Exatamente como a V3 espera)
target_col = 'NObeyesdad' if 'NObeyesdad' in df.columns else 'Obesity'
if 'family_history_with_overweight' in df.columns:
    df = df.rename(columns={'family_history_with_overweight': 'family_history'})

# 3. Criar a coluna 'Risk_Interaction' (O nome que a V3 usa)
df['Risk_Interaction'] = df['Weight'] / (df['Height'] ** 2)

X = df.drop(columns=[target_col])
y = df[target_col]

# 4. Definir Colunas (Sincronizado com input_data da V3)
categorical_features = ['Gender', 'family_history', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
numerical_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'Risk_Interaction']

# 5. Criar Pipeline Completo (Isso evita o erro de "Feature names unseen")
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# O segredo é salvar o Pipeline TODO (Processamento + IA)
clf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))
])

print("⚙️ Treinando IA robusta...")
clf_pipeline.fit(X, y)

# 6. Salvar os DOIS arquivos com o mesmo objeto (O Pipeline completo)
joblib.dump(clf_pipeline, 'best_obesity_model.pkl')
joblib.dump(clf_pipeline, 'full_pipeline_v4.pkl')

print("🎉 SUCESSO! O modelo agora entende a Lógica V3.")