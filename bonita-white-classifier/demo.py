# -*- coding: utf-8 -*-
"""
Bonita White AI - Demo para C-Level Executives
Aplicacion Streamlit para demostracion del modelo de clasificacion de cultivos

Autor: Computer Vision Team
Fecha: 2026
"""

import streamlit as st
import torch
import numpy as np
import cv2
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Configuracion de pagina
st.set_page_config(
    page_title="Bonita White AI | Clasificacion Fenologica",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from models.efficientnet_model import EfficientNetClassifier

    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

# ============================================
# CONFIGURACION
# ============================================
CLASS_NAMES = {
    0: "Estado 0 - Prefloracion",
    1: "Estado 1 - Floracion Intermedia",
    2: "Estado 2 - Floracion Maxima",
}

CLASS_COLORS = {0: "#34d399", 1: "#fbbf24", 2: "#f472b6"}

CLASS_DESCRIPTIONS = {
    0: "Dias 1-4: Campo predominantemente verde, pocas flores visibles",
    1: "Dias 5-8: Cobertura intermedia 40-60% de flores blancas",
    2: "Dias 9-11: Floracion maxima 80-90%, listo para corte",
}


# ============================================
# FUNCIONES DEL MODELO
# ============================================
@st.cache_resource
def load_model():
    """Cargar modelo entrenado"""
    try:
        model, info = EfficientNetClassifier.load_checkpoint(
            "checkpoints/best_model.pth", device="cpu"
        )
        model.eval()
        return model, info
    except Exception as e:
        st.error(f"Error cargando modelo: {e}")
        return None, None


def preprocess_image(image):
    """Preprocesar imagen para el modelo"""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (224, 224))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image / 255.0
    image = (image - mean) / std

    image = torch.from_numpy(image).permute(2, 0, 1).float()
    image = image.unsqueeze(0)

    return image


def predict(model, image_tensor):
    """Realizar prediccion"""
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        probs = probabilities[0].numpy()

    return predicted_class, probs


# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.title("🌾 Bonita White AI")
    st.markdown("---")

    page = st.radio(
        "Navegacion",
        ["Dashboard", "Demo Clasificacion", "Metricas del Modelo", "Informacion"],
    )

    st.markdown("---")
    st.markdown("### Estadisticas del Proyecto")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Videos", "14")
    with col2:
        st.metric("Frames", "761")

    col3, col4 = st.columns(2)
    with col3:
        st.metric("Accuracy", "90.5%")
    with col4:
        st.metric("Clases", "3")

# ============================================
# DASHBOARD
# ============================================
if page == "Dashboard":
    st.title("🌾 Bonita White AI")
    st.subheader("Sistema Inteligente de Clasificacion Fenologica")

    # KPIs
    st.markdown("### KPIs del Proyecto")

    kpi_cols = st.columns(4)
    metrics = [
        ("Accuracy", "90.5%", "#2dd4bf"),
        ("Frames", "761", "#34d399"),
        ("Videos", "14", "#fbbf24"),
        ("Clases", "3", "#f472b6"),
    ]

    for col, (name, value, color) in zip(kpi_cols, metrics):
        with col:
            st.markdown(
                f"""
            <div style="background: rgba(30, 41, 59, 0.8); border-radius: 16px; padding: 1.5rem; border-left: 4px solid {color}; margin-bottom: 1rem;">
                <h4 style="color: #94a3b8; margin: 0;">{name}</h4>
                <h2 style="color: {color}; margin: 0.5rem 0;">{value}</h2>
            </div>
            """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # Graficos
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Distribucion del Dataset")

        df_dist = pd.DataFrame(
            {
                "Estado": [
                    "Estado 0 (Dias 1-4)",
                    "Estado 1 (Dias 5-8)",
                    "Estado 2 (Dias 9-11)",
                ],
                "Frames": [315, 254, 192],
            }
        )

        fig_dist = px.bar(
            df_dist,
            x="Estado",
            y="Frames",
            color="Estado",
            color_discrete_sequence=["#34d399", "#fbbf24", "#f472b6"],
            text="Frames",
        )
        fig_dist.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#cbd5e1",
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_right:
        st.markdown("### Precision por Estado")

        df_acc = pd.DataFrame(
            {
                "Estado": ["Estado 0", "Estado 1", "Estado 2"],
                "Precision": [100, 78, 100],
                "Recall": [100, 100, 62],
            }
        )

        fig_acc = go.Figure()
        fig_acc.add_trace(
            go.Bar(
                name="Precision",
                x=df_acc["Estado"],
                y=df_acc["Precision"],
                marker_color="#2dd4bf",
            )
        )
        fig_acc.add_trace(
            go.Bar(
                name="Recall",
                x=df_acc["Estado"],
                y=df_acc["Recall"],
                marker_color="#34d399",
            )
        )

        fig_acc.update_layout(
            barmode="group",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#cbd5e1",
        )
        st.plotly_chart(fig_acc, use_container_width=True)

# ============================================
# DEMO CLASIFICACION
# ============================================
elif page == "Demo Clasificacion":
    st.title("🔮 Demo de Clasificacion")
    st.subheader("Sube una imagen del cultivo y obten la prediccion")

    model, info = load_model()

    if model is None:
        st.error("No se pudo cargar el modelo. Verifica que el checkpoint exista.")
    else:
        st.success("✅ Modelo cargado correctamente")

        uploaded_file = st.file_uploader(
            "Sube una imagen del cultivo (JPG, PNG)", type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            col_img, col_result = st.columns([1, 1])

            with col_img:
                st.markdown("### Imagen Original")
                st.image(
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True
                )

            with col_result:
                image_tensor = preprocess_image(image)
                predicted_class, probabilities = predict(model, image_tensor)
                confidence = probabilities[predicted_class] * 100

                st.markdown("### Prediccion")

                st.markdown(
                    f"""
                <div style="background: linear-gradient(135deg, rgba(45, 212, 191, 0.1) 0%, rgba(52, 211, 153, 0.1) 100%); border-radius: 20px; padding: 2rem; border: 2px solid rgba(45, 212, 191, 0.3); text-align: center;">
                    <h2 style="color: {CLASS_COLORS[predicted_class]}; margin: 0;">
                        {CLASS_NAMES[predicted_class]}
                    </h2>
                    <h1 style="font-size: 4rem; margin: 1rem 0; color: #f1f5f9;">
                        {confidence:.1f}%
                    </h1>
                    <p style="color: #94a3b8; font-size: 1.1rem;">
                        Confianza de la prediccion
                    </p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                st.info(CLASS_DESCRIPTIONS[predicted_class])

            st.markdown("---")
            st.markdown("### Distribucion de Probabilidades")

            prob_df = pd.DataFrame(
                {
                    "Estado": list(CLASS_NAMES.values()),
                    "Probabilidad": probabilities * 100,
                }
            )

            fig_prob = px.bar(
                prob_df,
                x="Estado",
                y="Probabilidad",
                color="Estado",
                color_discrete_map={
                    CLASS_NAMES[0]: CLASS_COLORS[0],
                    CLASS_NAMES[1]: CLASS_COLORS[1],
                    CLASS_NAMES[2]: CLASS_COLORS[2],
                },
            )

            fig_prob.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#cbd5e1",
                showlegend=False,
            )

            st.plotly_chart(fig_prob, use_container_width=True)

# ============================================
# METRICAS
# ============================================
elif page == "Metricas del Modelo":
    st.title("📊 Metricas del Modelo")
    st.subheader("Rendimiento detallado del clasificador")

    st.markdown("### Metricas Globales (Test Set)")

    metrics_cols = st.columns(4)
    metrics_data = [
        ("Accuracy", "90.52%", "#2dd4bf"),
        ("Precision", "92.67%", "#34d399"),
        ("Recall", "87.36%", "#fbbf24"),
        ("F1-Score", "88.08%", "#f472b6"),
    ]

    for col, (name, value, color) in zip(metrics_cols, metrics_data):
        with col:
            st.markdown(
                f"""
            <div style="background: rgba(30, 41, 59, 0.8); border-radius: 16px; padding: 1.5rem; border-left: 4px solid {color}; margin-bottom: 1rem;">
                <h4 style="color: #94a3b8; margin: 0;">{name}</h4>
                <h2 style="color: {color}; margin: 0.5rem 0;">{value}</h2>
            </div>
            """,
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("### Reporte por Clase")

    report_data = {
        "Estado": [
            "Estado 0 - Prefloracion",
            "Estado 1 - Floracion Intermedia",
            "Estado 2 - Floracion Maxima",
        ],
        "Precision": ["100%", "78%", "100%"],
        "Recall": ["100%", "100%", "62%"],
        "F1-Score": ["100%", "88%", "77%"],
        "Muestras": [48, 39, 29],
    }

    df_report = pd.DataFrame(report_data)
    st.dataframe(df_report, use_container_width=True, hide_index=True)

# ============================================
# INFORMACION
# ============================================
elif page == "Informacion":
    st.title("ℹ️ Informacion del Proyecto")

    st.markdown("""
    ### Objetivo del Proyecto
    
    Desarrollar un sistema de inteligencia artificial capaz de clasificar automaticamente 
    el estado fenologico del cultivo Bonita White mediante analisis de imagenes aereas.
    
    ### Arquitectura del Modelo
    
    - **Backbone**: EfficientNet-B0 (Transfer Learning)
    - **Input**: Imagenes RGB 224x224 pixeles
    - **Output**: 3 clases de estado fenologico
    - **Framework**: PyTorch 2.0
    
    ### Dataset
    
    | Fuente | Cantidad |
    |--------|----------|
    | Videos procesados | 14 |
    | Frames extraidos | 761 |
    | Dias de monitoreo | 11 |
    
    ### Beneficios para el Negocio
    
    - **Automatizacion**: Elimina inspeccion manual
    - **Escala**: Analiza miles de hectareas en minutos
    - **Precision**: 90.5% de accuracy en clasificacion
    - **ROI**: Optimiza timing de corte para maxima calidad
    """)

st.markdown("---")
st.markdown(
    """
<p style="text-align: center; color: #64748b;">
    Bonita White AI 2026 | Computer Vision Team
</p>
""",
    unsafe_allow_html=True,
)
