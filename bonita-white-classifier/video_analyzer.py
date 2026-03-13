# -*- coding: utf-8 -*-
"""
Bonita White AI - Analisis de Videos Completos

Aplicacion Streamlit para analisis frame-by-frame de videos de cultivos.
Genera reportes completos con estadisticas temporales y estado predominante.

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
import tempfile
import time
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from models.efficientnet_model import EfficientNetClassifier

    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

# Configuracion de pagina
st.set_page_config(
    page_title="Bonita White AI | Analisis de Videos",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================
# CONFIGURACION
# ============================================
CLASS_NAMES = {
    0: "Estado 0 - Prefloracion",
    1: "Estado 1 - Floracion Intermedia",
    2: "Estado 2 - Floracion Maxima",
}

CLASS_COLORS = {0: "#34d399", 1: "#fbbf24", 2: "#f472b6"}

CLASS_SHORT = {0: "Estado 0", 1: "Estado 1", 2: "Estado 2"}


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


def preprocess_frame(frame):
    """Preprocesar frame para el modelo"""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    frame = frame / 255.0
    frame = (frame - mean) / std

    frame = torch.from_numpy(frame).permute(2, 0, 1).float()
    frame = frame.unsqueeze(0)

    return frame


def predict_frame(model, frame_tensor):
    """Realizar prediccion en un frame"""
    with torch.no_grad():
        outputs = model(frame_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        probs = probabilities[0].numpy()

    return predicted_class, probs


# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.title("🎥 Analisis de Videos")
    st.markdown("---")

    st.markdown("### 📊 Que analizamos?")
    st.markdown("""
    - ✅ Clasificacion frame por frame
    - ✅ Estado predominante del video
    - ✅ Timeline de transiciones
    - ✅ Estadisticas temporales
    - ✅ Reporte ejecutivo completo
    """)

    st.markdown("---")
    st.markdown("### 🎯 Estados del Cultivo")
    for i, (name, color) in enumerate(zip(CLASS_NAMES.values(), CLASS_COLORS.values())):
        st.markdown(
            f"<span style='color: {color}'>●</span> {name}", unsafe_allow_html=True
        )

# ============================================
# MAIN CONTENT
# ============================================
st.title("🎥 Analisis de Videos Completos")
st.subheader("Clasificacion frame-by-frame con reporte ejecutivo")

# Verificar modelo
model, info = load_model()

if model is None:
    st.error(
        "❌ No se pudo cargar el modelo. Verifica que exista 'checkpoints/best_model.pth'"
    )
    st.stop()
else:
    st.success("✅ Modelo cargado correctamente")

# Upload de video
uploaded_video = st.file_uploader(
    "📤 Sube un video del cultivo (MP4, AVI, MOV)",
    type=["mp4", "avi", "mov"],
    help="El video sera analizado frame por frame para determinar el estado fenologico",
)

if uploaded_video is not None:
    # Guardar video temporal
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    # Abrir video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0

    # Mostrar info del video
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric("Duracion", f"{duration:.1f} seg")
    with col_info2:
        st.metric("Total Frames", f"{total_frames}")
    with col_info3:
        st.metric("FPS", f"{fps:.1f}")

    # Configuracion de analisis
    st.markdown("---")
    st.markdown("### ⚙️ Configuracion de Analisis")

    col_conf1, col_conf2 = st.columns(2)
    with col_conf1:
        frame_interval = st.slider(
            "Intervalo de analisis (frames)",
            min_value=1,
            max_value=30,
            value=5,
            help="Analizar 1 de cada N frames. Mayor intervalo = analisis mas rapido",
        )
    with col_conf2:
        confidence_threshold = (
            st.slider(
                "Umbral de confianza (%)",
                min_value=50,
                max_value=95,
                value=70,
                help="Solo considerar predicciones con confianza mayor a este valor",
            )
            / 100
        )

    # Boton de analisis
    if st.button(
        "🚀 Iniciar Analisis del Video", type="primary", use_container_width=True
    ):
        # Preparar para analisis
        frames_to_process = total_frames // frame_interval

        progress_bar = st.progress(0)
        status_text = st.empty()

        # Almacenar resultados
        results = []
        frame_indices = []
        timestamps = []

        # Procesar video
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        st.markdown("### 📊 Procesando Video...")

        # Mostrar preview en tiempo real
        preview_col, stats_col = st.columns([2, 1])

        with preview_col:
            frame_display = st.empty()
            prediction_display = st.empty()

        with stats_col:
            current_state = st.empty()
            confidence_display = st.empty()
            frames_processed = st.empty()

        frame_count = 0
        processed_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Procesar solo cada N frames
            if frame_count % frame_interval == 0:
                # Preprocesar y predecir
                frame_tensor = preprocess_frame(frame)
                predicted_class, probabilities = predict_frame(model, frame_tensor)
                confidence = probabilities[predicted_class]

                timestamp = frame_count / fps if fps > 0 else 0

                # Guardar resultado
                results.append(
                    {
                        "frame": frame_count,
                        "timestamp": timestamp,
                        "class": predicted_class,
                        "class_name": CLASS_NAMES[predicted_class],
                        "confidence": confidence,
                        "prob_estado_0": probabilities[0],
                        "prob_estado_1": probabilities[1],
                        "prob_estado_2": probabilities[2],
                    }
                )

                frame_indices.append(frame_count)
                timestamps.append(timestamp)
                processed_count += 1

                # Actualizar preview (cada 10 frames para no relentizar)
                if processed_count % 5 == 0 or processed_count == 1:
                    # Redimensionar frame para display
                    display_frame = cv2.resize(frame, (480, 270))
                    display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

                    with preview_col:
                        frame_display.image(
                            display_frame, channels="RGB", use_container_width=True
                        )

                        # Mostrar prediccion actual
                        pred_color = CLASS_COLORS[predicted_class]
                        prediction_display.markdown(
                            f"""
                        <div style="background: {pred_color}30; border-left: 4px solid {pred_color}; padding: 10px; border-radius: 5px; margin-top: 10px;">
                            <h4 style="margin: 0; color: {pred_color};">{CLASS_NAMES[predicted_class]}</h4>
                            <p style="margin: 5px 0 0 0; color: #94a3b8;">Confianza: {confidence * 100:.1f}%</p>
                            <p style="margin: 5px 0 0 0; color: #64748b; font-size: 0.9rem;">Tiempo: {timestamp:.1f}s (Frame {frame_count})</p>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )

                    with stats_col:
                        current_state.markdown(
                            f"**Estado Actual:**\n<h3 style='color: {pred_color};'>{CLASS_SHORT[predicted_class]}</h3>",
                            unsafe_allow_html=True,
                        )
                        confidence_display.progress(
                            float(confidence),
                            text=f"Confianza: {confidence * 100:.1f}%",
                        )
                        frames_processed.metric(
                            "Frames Analizados",
                            f"{processed_count}/{frames_to_process}",
                        )

                # Actualizar barra de progreso
                progress = min(1.0, processed_count / frames_to_process)
                progress_bar.progress(progress)
                status_text.text(
                    f"Procesando... {processed_count}/{frames_to_process} frames ({progress * 100:.1f}%)"
                )

            frame_count += 1

        cap.release()

        st.success(f"✅ Analisis completado! {processed_count} frames procesados")

        # ============================================
        # REPORTE EJECUTIVO
        # ============================================
        st.markdown("---")
        st.markdown("## 📈 Reporte Ejecutivo del Video")

        df_results = pd.DataFrame(results)

        # Calcular estadisticas
        class_counts = df_results["class"].value_counts().to_dict()
        total_analyzed = len(df_results)

        # Estado predominante
        predominant_class = max(class_counts.items(), key=lambda x: x[1])[0]
        predominant_pct = (class_counts[predominant_class] / total_analyzed) * 100

        # KPIs principales
        kpi_cols = st.columns(4)
        with kpi_cols[0]:
            st.markdown(
                f"""
            <div style="background: {CLASS_COLORS[predominant_class]}20; border-radius: 12px; padding: 20px; border: 2px solid {CLASS_COLORS[predominant_class]}; text-align: center;">
                <h4 style="color: #94a3b8; margin: 0;">Estado Predominante</h4>
                <h2 style="color: {CLASS_COLORS[predominant_class]}; margin: 10px 0;">{CLASS_SHORT[predominant_class]}</h2>
                <p style="color: #64748b; margin: 0;">{predominant_pct:.1f}% del video</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with kpi_cols[1]:
            avg_confidence = df_results["confidence"].mean() * 100
            st.metric("Confianza Promedio", f"{avg_confidence:.1f}%")

        with kpi_cols[2]:
            st.metric("Frames Analizados", f"{total_analyzed}")

        with kpi_cols[3]:
            st.metric("Duracion Total", f"{duration:.1f}s")

        st.markdown("---")

        # Distribucion de estados
        col_chart1, col_chart2 = st.columns([2, 1])

        with col_chart1:
            st.markdown("### 📊 Distribucion de Estados en el Video")

            # Timeline de clasificaciones
            fig_timeline = go.Figure()

            for class_id in [0, 1, 2]:
                mask = df_results["class"] == class_id
                if mask.any():
                    fig_timeline.add_trace(
                        go.Scatter(
                            x=df_results[mask]["timestamp"],
                            y=[class_id] * mask.sum(),
                            mode="markers",
                            name=CLASS_SHORT[class_id],
                            marker=dict(
                                color=CLASS_COLORS[class_id], size=10, opacity=0.7
                            ),
                        )
                    )

            fig_timeline.update_layout(
                title="Timeline de Clasificacion (Tiempo vs Estado)",
                xaxis_title="Tiempo (segundos)",
                yaxis_title="Estado",
                yaxis=dict(
                    tickmode="array",
                    tickvals=[0, 1, 2],
                    ticktext=["Estado 0", "Estado 1", "Estado 2"],
                ),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#cbd5e1",
                height=400,
            )

            st.plotly_chart(fig_timeline, use_container_width=True)

        with col_chart2:
            st.markdown("### 📈 Porcentaje por Estado")

            # Pie chart
            pie_data = []
            pie_labels = []
            pie_colors = []

            for class_id in sorted(class_counts.keys()):
                pct = (class_counts[class_id] / total_analyzed) * 100
                pie_data.append(pct)
                pie_labels.append(f"{CLASS_SHORT[class_id]}\n{pct:.1f}%")
                pie_colors.append(CLASS_COLORS[class_id])

            fig_pie = go.Figure(
                data=[
                    go.Pie(
                        labels=pie_labels,
                        values=pie_data,
                        marker_colors=pie_colors,
                        hole=0.4,
                        textinfo="label",
                    )
                ]
            )

            fig_pie.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#cbd5e1",
                showlegend=False,
                height=400,
            )

            st.plotly_chart(fig_pie, use_container_width=True)

        # Analisis de transiciones
        st.markdown("---")
        st.markdown("### 🔄 Analisis de Transiciones")

        # Detectar transiciones
        transitions = []
        for i in range(1, len(df_results)):
            if df_results.iloc[i]["class"] != df_results.iloc[i - 1]["class"]:
                transitions.append(
                    {
                        "timestamp": df_results.iloc[i]["timestamp"],
                        "from": CLASS_SHORT[df_results.iloc[i - 1]["class"]],
                        "to": CLASS_SHORT[df_results.iloc[i]["class"]],
                    }
                )

        if transitions:
            st.write(f"**Transiciones detectadas:** {len(transitions)}")

            trans_df = pd.DataFrame(transitions)
            st.dataframe(trans_df, use_container_width=True, hide_index=True)
        else:
            st.info("No se detectaron transiciones entre estados en este video")

        # Tabla de resultados detallada
        st.markdown("---")
        st.markdown("### 📋 Datos Detallados")

        with st.expander("Ver tabla completa de resultados"):
            st.dataframe(df_results, use_container_width=True)

            # Boton de descarga
            csv = df_results.to_csv(index=False)
            st.download_button(
                label="📥 Descargar CSV",
                data=csv,
                file_name=f"analisis_{uploaded_video.name}.csv",
                mime="text/csv",
            )

        # Resumen ejecutivo
        st.markdown("---")
        st.markdown("### 📝 Resumen Ejecutivo")

        resumen = f"""
        **Video analizado:** {uploaded_video.name}  
        **Duracion:** {duration:.1f} segundos ({total_frames} frames totales)  
        **Estado predominante:** {CLASS_NAMES[predominant_class]} ({predominant_pct:.1f}% del tiempo)  
        **Confianza promedio:** {avg_confidence:.1f}%  
        **Transiciones detectadas:** {len(transitions)}  
        
        **Recomendacion:** Este video muestra principalmente un cultivo en **{CLASS_NAMES[predominant_class]}**. 
        """

        if predominant_class == 0:
            resumen += "El campo aun no esta listo para corte. Se recomienda continuar monitoreo."
        elif predominant_class == 1:
            resumen += "El cultivo esta en fase intermedia. Evaluar en 2-3 dias para determinar momento optimo de corte."
        else:
            resumen += "El cultivo esta en floracion maxima. Es el momento optimo para programar el corte."

        st.markdown(resumen)

st.markdown("---")
st.markdown(
    """
<p style="text-align: center; color: #64748b;">
    Bonita White AI 2026 | Analisis de Videos | Computer Vision Team
</p>
""",
    unsafe_allow_html=True,
)
