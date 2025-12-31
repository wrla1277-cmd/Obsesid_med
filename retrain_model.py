import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

print("🧠 Iniciando treinamento AVANÇADO de Inteligência Clínica (V5)...")

# 1. Carregar Dados
try:
    df = pd.read_csv('Obesity.csv')
    # Tratamento de separador caso necessário
    if len(df.columns) < 2: df = pd.read_csv('Obesity.csv', sep=';')
except Exception as e:
    print(f"❌ Erro ao ler CSV: {e}")
    exit()

# 2. Preparação dos Dados
# IMPORTANTE: Não criamos a coluna 'Risk_Interaction' (IMC).
# Deixamos o modelo descobrir as relações sozinho.

target_col = 'NObeyesdad'
if target_col not in df.columns: 
    target_col = 'Obesity' # Fallback para outros nomes comuns

# Normalização de nomes de colunas
if 'family_history_with_overweight' in df.columns:
    df = df.rename(columns={'family_history_with_overweight': 'family_history'})

X = df.drop(columns=[target_col])
y = df[target_col]

# 3. Definição de Features (Foco Multidimensional)
# Removemos Risk_Interaction e mantemos o foco nos hábitos + biometria pura
categorical_features = ['Gender', 'family_history', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
numerical_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

# Validação se as colunas existem
missing_cols = [col for col in categorical_features + numerical_features if col not in X.columns]
if missing_cols:
    print(f"⚠️ Aviso: Colunas faltando no CSV: {missing_cols}")

# 4. Pipeline de Processamento Robusto
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 5. Modelo (Random Forest Otimizado)
# max_depth=15 evita overfitting (decorar o peso exato)
# class_weight='balanced' ajuda a não ignorar classes menores
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=300, 
        max_depth=15, 
        random_state=42, 
        class_weight='balanced',
        n_jobs=-1
    ))
])

# 6. Treinamento
print("⚙️  Treinando a IA para cruzar Hábitos x Biometria...")
model_pipeline.fit(X, y)

# 7. Prova Real: O que a IA está olhando?
print("\n📊 FATORES MAIS IMPORTANTES PARA A DECISÃO DA IA:")
try:
    # Extração técnica para mostrar a importância das features
    feature_names_num = numerical_features
    feature_names_cat = model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
    all_features = np.r_[feature_names_num, feature_names_cat]
    
    importances = model_pipeline.named_steps['classifier'].feature_importances_
    
    df_imp = pd.DataFrame({'Fator': all_features, 'Importancia': importances})
    print(df_imp.sort_values('Importancia', ascending=False).head(10).to_string(index=False))
except Exception as e:
    print(f"(Não foi possível gerar a tabela de importâncias: {e})")

# 8. Salvar os Cérebro Novo
joblib.dump(model_pipeline, 'best_obesity_model.pkl')
joblib.dump(model_pipeline, 'full_pipeline_v4.pkl') # Mantendo nome para compatibilidade
print("\n✅ SUCESSO! Novo modelo focado em comportamento salvo.")