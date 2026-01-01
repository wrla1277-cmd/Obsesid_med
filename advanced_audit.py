import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve

# Configurações de Design
COLORS = {
    'primary': '#1e3a8a',    # Azul profundo
    'secondary': '#3b82f6',  # Azul vibrante
    'accent': '#10b981',     # Verde esmeralda
    'background': '#f8fafc', # Cinza muito claro
    'text': '#1e293b',       # Cinza escuro
    'grid': '#e2e8f0',       # Cinza claro para grid
    'classes': ['#1e3a8a', '#3b82f6', '#60a5fa', '#93c5fd', '#bfdbfe', '#dbeafe', '#eff6ff']
}

CLASS_MAP = {
    'Insufficient_Weight': 'Peso Insuficiente', 
    'Normal_Weight': 'Peso Normal',
    'Overweight_Level_I': 'Sobrepeso I', 
    'Overweight_Level_II': 'Sobrepeso II',
    'Obesity_Type_I': 'Obesidade I', 
    'Obesity_Type_II': 'Obesidade II', 
    'Obesity_Type_III': 'Obesidade III'
}

def generate_audit():
    print("🚀 Iniciando Auditoria Visual de Alta Performance...")
    
    # Carregamento de dados
    model = joblib.load('best_obesity_model.pkl')
    df = pd.read_csv('Obesity.csv')
    
    target = 'NObeyesdad' if 'NObeyesdad' in df.columns else 'Obesity'
    X = df.drop(columns=[target])
    y = df[target]
    
    labels_raw = sorted(y.unique())
    labels_pt = [CLASS_MAP.get(l, l) for l in labels_raw]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)
    
    # Criar Dashboard com Subplots (3x2)
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "<b>1. Matriz de Diagnóstico (Real vs IA)</b>", 
            "<b>2. Top 10 Fatores Determinantes</b>",
            "<b>3. Curvas de Sensibilidade (ROC)</b>", 
            "<b>4. Evolução do Aprendizado</b>",
            "<b>5. Calibração de Confiança Clínica</b>", 
            "<b>6. Equilíbrio Precisão-Recall (Visão Bônus)</b>"
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    # ------------------------------------------------------------------------------
    # 1. MATRIZ DE CONFUSÃO (Heatmap Moderno)
    # ------------------------------------------------------------------------------
    cm = confusion_matrix(y_test, y_pred, labels=labels_raw)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig.add_trace(
        go.Heatmap(
            z=cm,
            x=labels_pt,
            y=labels_pt,
            colorscale='Blues',
            showscale=False,
            text=cm,
            texttemplate="%{text}",
            hoverinfo='z'
        ),
        row=1, col=1
    )
    fig.update_xaxes(title_text="Previsão da IA", row=1, col=1)
    fig.update_yaxes(title_text="Realidade Clínica", row=1, col=1)

    # ------------------------------------------------------------------------------
    # 2. FEATURE IMPORTANCE (Bar Chart Elegante)
    # ------------------------------------------------------------------------------
    classifier = model.named_steps['classifier']
    preprocessor = model.named_steps['preprocessor']
    
    # Extração dinâmica de nomes de features
    num_features = ['Idade', 'Altura', 'Peso', 'Vegetais', 'Refeições', 'Água', 'Exercício', 'Telas']
    cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out()
    all_feat_names = np.concatenate([num_features, cat_features])
    
    feat_imp = pd.Series(classifier.feature_importances_, index=all_feat_names).sort_values(ascending=True).tail(10)
    
    fig.add_trace(
        go.Bar(
            x=feat_imp.values,
            y=feat_imp.index,
            orientation='h',
            marker_color=COLORS['secondary'],
            showlegend=False
        ),
        row=1, col=2
    )

    # ------------------------------------------------------------------------------
    # 3. CURVA ROC (Interativa e Limpa)
    # ------------------------------------------------------------------------------
    y_test_bin = label_binarize(y_test, classes=labels_raw)
    
    for i in range(len(labels_raw)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        fig.add_trace(
            go.Scatter(x=fpr, y=tpr, name=f"{labels_pt[i]} (AUC={roc_auc:.2f})", mode='lines'),
            row=2, col=1
        )
    
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash='dash', color='grey'), showlegend=False),
        row=2, col=1
    )
    fig.update_xaxes(title_text="Taxa Falso Positivo", row=2, col=1)
    fig.update_yaxes(title_text="Taxa Verdadeiro Positivo", row=2, col=1)

    # ------------------------------------------------------------------------------
    # 4. CURVA DE APRENDIZADO
    # ------------------------------------------------------------------------------
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=3, n_jobs=-1)
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    
    fig.add_trace(
        go.Scatter(x=train_sizes, y=train_mean, name="Treino", line=dict(color=COLORS['primary'])),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=train_sizes, y=test_mean, name="Validação", line=dict(color=COLORS['accent'])),
        row=2, col=2
    )
    fig.update_xaxes(title_text="Volume de Dados", row=2, col=2)
    fig.update_yaxes(title_text="Acurácia", row=2, col=2)

    # ------------------------------------------------------------------------------
    # 5. CALIBRAÇÃO (Foco na Classe Crítica: Obesidade III)
    # ------------------------------------------------------------------------------
    critical_idx = labels_raw.index('Obesity_Type_III')
    y_test_cal = (y_test == labels_raw[critical_idx]).astype(int)
    prob_pos = y_score[:, critical_idx]
    frac_pos, mean_pred = calibration_curve(y_test_cal, prob_pos, n_bins=10)
    
    fig.add_trace(
        go.Scatter(x=mean_pred, y=frac_pos, name="Calibração IA", mode='lines+markers', marker=dict(symbol='square')),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash='dash', color='black'), name="Referência"),
        row=3, col=1
    )
    fig.update_xaxes(title_text="Confiança da IA", row=3, col=1)
    fig.update_yaxes(title_text="Precisão Empírica", row=3, col=1)

    # ------------------------------------------------------------------------------
    # 6. TAREFA BÔNUS: PRECISION-RECALL CURVE
    # ------------------------------------------------------------------------------
    # Esta visão é crucial para datasets desbalanceados ou diagnósticos clínicos onde
    # o custo de falsos negativos é alto.
    for i in range(len(labels_raw)):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        fig.add_trace(
            go.Scatter(x=recall, y=precision, name=f"PR: {labels_pt[i]}", mode='lines', showlegend=False),
            row=3, col=2
        )
    fig.update_xaxes(title_text="Recall (Sensibilidade)", row=3, col=2)
    fig.update_yaxes(title_text="Precisão", row=3, col=2)

    # Layout Final
    fig.update_layout(
        height=1400,
        width=1200,
        title_text="<b>AUDITORIA TÉCNICA: MODELO DE CLASSIFICAÇÃO DE OBESIDADE</b><br><span style='font-size:14px'>Relatório Executivo de Performance e Confiabilidade Clínica</span>",
        title_x=0.5,
        template='plotly_white',
        font=dict(family="Arial, sans-serif", size=12, color=COLORS['text']),
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
        margin=dict(t=120, b=150, l=80, r=80)
    )

    # Salvar como HTML estático para visualização
    fig.write_html("dashboard_auditoria.html")
    
    # Salvar como PNG estático para visualização rápida
    try:
        fig.write_image("dashboard_auditoria.png", scale=2)
        print("✅ Dashboard gerado com sucesso: dashboard_auditoria.html e dashboard_auditoria.png")
    except Exception as e:
        print(f"⚠️ Erro ao salvar PNG: {e}")
        print("✅ Dashboard gerado com sucesso: dashboard_auditoria.html")

if __name__ == "__main__":
    generate_audit()
