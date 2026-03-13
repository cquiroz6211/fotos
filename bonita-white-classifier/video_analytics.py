# -*- coding: utf-8 -*-
"""
Bonita White AI - Video Analytics Dashboard
Dashboard Streamlit para analisis completo de videos de cultivo

Autor: Computer Vision Team
Fecha: 2026
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from inference.video_processor import create_video_processor, VideoProcessor

    VIDEO_PROCESSOR_AVAILABLE = True
except ImportError as e:
    VIDEO_PROCESSOR_AVAILABLE = False
    st.error(f"Error importando VideoProcessor: {e}")

# Configuracion de pagina
st.set_page_config(
    page_title="Bonita White AI - Video Analytics",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================
# ESTADOS DE CLASE
# ============================================
STATE_NAMES = {
    0: "Estado 0 - Prefloracion",
    1: "Estado 1 - Floracion Intermedia",
    2: "Estado 2 - Floracion Maxima",
}

STATE_COLORS = {
    0: "#34d399",  # Verde
    1: "#fbbf24",  # Amarillo
    2: "#f472b6",  # Rosa
}

STATE_DESCRIPTIONS = {
    0: "🌱 Dias 1-4: Campo predominantemente verde, pocas flores visibles",
    1: "🌼 Dias 5-8: Cobertura intermedia 40-60% de flores blancas",
    2: "🌸 Dias 9-11: Floracion maxima 80-90%, listo para corte",
}

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.title("🎬 Video Analytics")
    st.markdown("---")

    st.markdown("### Configuracion de Analisis")

    frame_interval = st.slider(
        "Intervalo de analisis (frames)",
        min_value=1,
        max_value=60,
        value=30,
        help="Analizar 1 frame cada N frames. 30 = ~1 frame/segundo en video a 30fps",
    )

    confidence_threshold = st.slider(
        "Umbral de confianza minima",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Predicciones con confianza menor a este valor seran marcadas como inciertas",
    )

    st.markdown("---")

    if VIDEO_PROCESSOR_AVAILABLE:
        st.success("✅ Sistema listo para analisis")
    else:
        st.error("❌ Error en el sistema")

    st.markdown("---")
    st.markdown("### 📊 Metricas del Modelo")
    st.metric("Accuracy", "90.5%")
    st.metric("Precision", "92.7%")
    st.metric("Recall", "87.4%")

# ============================================
# MAIN CONTENT
# ============================================
st.title("🎬 Analisis de Video - Bonita White AI")
st.subheader("Procesamiento completo de videos para clasificacion fenologica")

# Tabs para organizar la interfaz
tab_upload, tab_results, tab_timeline, tab_export = st.tabs(
    ["📤 Subir Video", "📊 Resultados", "📈 Timeline", "💾 Exportar"]
)

# ============================================
# TAB 1: UPLOAD
# ============================================
with tab_upload:
    st.markdown("### 📤 Subir Video para Analisis")

    uploaded_video = st.file_uploader(
        "Selecciona un video MP4 del cultivo",
        type=["mp4", "avi", "mov"],
        help="El video sera analizado frame por frame para determinar el estado fenologico",
    )

    if uploaded_video is not None:
        # Guardar video temporalmente
        temp_path = Path("temp_video.mp4")
        temp_path.write_bytes(uploaded_video.read())

        st.video(str(temp_path))

        # Boton de analisis
        if st.button("🚀 Iniciar Analisis", type="primary", use_container_width=True):
            if not VIDEO_PROCESSOR_AVAILABLE:
                st.error("❌ El procesador de video no esta disponible")
            else:
                # Crear processor
                try:
                    processor = create_video_processor(
                        checkpoint_path="checkpoints/best_model.pth",
                        device="cpu",
                        frame_interval=frame_interval,
                    )

                    # Procesar con barra de progreso
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    def update_progress(pct, current, total):
                        progress_bar.progress(int(pct))
                        status_text.text(
                            f"Procesando... {current}/{total} frames ({pct:.1f}%)"
                        )

                    with st.spinner(
                        "Analizando video... Esto puede tomar varios minutos"
                    ):
                        result = processor.process_video(
                            str(temp_path), progress_callback=update_progress
                        )

                    # Guardar en session state para otros tabs
                    st.session_state["video_result"] = result
                    st.session_state["processor"] = processor

                    st.success(
                        f"✅ Analisis completado! {result.processed_frames} frames procesados"
                    )

                    # Mostrar resumen inmediato
                    st.markdown("### 📋 Resumen del Analisis")

                    res_cols = st.columns(4)
                    with res_cols[0]:
                        st.metric("Duracion", f"{result.duration_seconds:.1f}s")
                    with res_cols[1]:
                        st.metric("Frames Analizados", result.processed_frames)
                    with res_cols[2]:
                        st.metric(
                            "Estado Predominante", f"Estado {result.dominant_state}"
                        )
                    with res_cols[3]:
                        st.metric("Confianza Promedio", f"{result.confidence_mean:.1%}")

                    st.info(f"**{STATE_DESCRIPTIONS[result.dominant_state]}**")

                except Exception as e:
                    st.error(f"❌ Error procesando video: {e}")
                    import traceback

                    st.error(traceback.format_exc())

# ============================================
# TAB 2: RESULTS
# ============================================
with tab_results:
    if "video_result" not in st.session_state:
        st.info("📤 Sube un video en la pestaña 'Subir Video' para ver los resultados")
    else:
        result = st.session_state["video_result"]

        st.markdown("### 📊 Resultados del Analisis")

        # KPIs
        kpi_cols = st.columns(4)
        with kpi_cols[0]:
            st.markdown(
                f"""
            <div style="background: rgba(30, 41, 59, 0.8); border-radius: 16px; padding: 1.5rem; text-align: center;">
                <h4 style="color: #94a3b8; margin: 0;">Duracion</h4>
                <h2 style="color: #2dd4bf; margin: 0.5rem 0;">{result.duration_seconds:.1f}s</h2>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with kpi_cols[1]:
            st.markdown(
                f"""
            <div style="background: rgba(30, 41, 59, 0.8); border-radius: 16px; padding: 1.5rem; text-align: center;">
                <h4 style="color: #94a3b8; margin: 0;">Frames Analizados</h4>
                <h2 style="color: #34d399; margin: 0.5rem 0;">{result.processed_frames}</h2>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with kpi_cols[2]:
            dominant_name = STATE_NAMES[result.dominant_state].split(" - ")[1]
            st.markdown(
                f"""
            <div style="background: rgba(30, 41, 59, 0.8); border-radius: 16px; padding: 1.5rem; text-align: center;">
                <h4 style="color: #94a3b8; margin: 0;">Estado Predominante</h4>
                <h2 style="color: {STATE_COLORS[result.dominant_state]}; margin: 0.5rem 0;">{dominant_name}</h2>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with kpi_cols[3]:
            st.markdown(
                f"""
            <div style="background: rgba(30, 41, 59, 0.8); border-radius: 16px; padding: 1.5rem; text-align: center;">
                <h4 style="color: #94a3b8; margin: 0;">Confianza</h4>
                <h2 style="color: #fbbf24; margin: 0.5rem 0;">{result.confidence_mean:.1%}</h2>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # Distribucion de estados
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🥧 Distribucion de Estados")

            dist_df = pd.DataFrame(
                {
                    "Estado": [
                        STATE_NAMES[s] for s in result.state_distribution.keys()
                    ],
                    "Porcentaje": list(result.state_distribution.values()),
                    "Color": [
                        STATE_COLORS[s] for s in result.state_distribution.keys()
                    ],
                }
            )

            fig_pie = px.pie(
                dist_df,
                values="Porcentaje",
                names="Estado",
                color="Estado",
                color_discrete_map={
                    row["Estado"]: row["Color"] for _, row in dist_df.iterrows()
                },
            )
            fig_pie.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#cbd5e1",
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            st.markdown("### 📊 Estadisticas de Confianza")

            conf_data = {
                "Metrica": ["Promedio", "Desviacion Estandar", "Minima", "Maxima"],
                "Valor": [
                    f"{result.confidence_mean:.2%}",
                    f"{result.confidence_std:.2%}",
                    f"{min([p.confidence for p in result.predictions]):.2%}",
                    f"{max([p.confidence for p in result.predictions]):.2%}",
                ],
            }
            st.table(pd.DataFrame(conf_data))

            # Estado predominante destacado
            st.markdown(
                f"""
            <div style="background: linear-gradient(135deg, rgba(45, 212, 191, 0.1) 0%, rgba(52, 211, 153, 0.1) 100%); 
                        border-radius: 16px; padding: 1.5rem; border: 2px solid rgba(45, 212, 191, 0.3); margin-top: 1rem;">
                <h4 style="color: #2dd4bf; margin: 0;">📌 Conclusion del Analisis</h4>
                <p style="color: #cbd5e1; margin: 0.5rem 0;">
                    El video muestra predominantemente <strong>{STATE_NAMES[result.dominant_state]}</strong> 
                    durante el {result.state_distribution[result.dominant_state]:.1f}% del tiempo analizado.
                </p>
                <p style="color: #94a3b8; margin: 0; font-size: 0.9rem;">
                    {STATE_DESCRIPTIONS[result.dominant_state]}
                </p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Transiciones detectadas
        if result.state_transitions:
            st.markdown("---")
            st.markdown("### 🔄 Transiciones de Estado Detectadas")

            trans_data = []
            for timestamp, from_state, to_state in result.state_transitions:
                time_str = str(timedelta(seconds=int(timestamp)))
                trans_data.append(
                    {
                        "Tiempo": time_str,
                        "Desde": STATE_NAMES[from_state],
                        "Hacia": STATE_NAMES[to_state],
                    }
                )

            st.table(pd.DataFrame(trans_data))

# ============================================
# TAB 3: TIMELINE
# ============================================
with tab_timeline:
    if "video_result" not in st.session_state:
        st.info("📤 Sube un video para ver el timeline de analisis")
    else:
        result = st.session_state["video_result"]

        st.markdown("### 📈 Timeline del Analisis")

        # Timeline de estados
        fig_timeline = px.scatter(
            result.timeline_data,
            x="timestamp",
            y="predicted_state",
            color="state_name",
            color_discrete_map={
                STATE_NAMES[0]: STATE_COLORS[0],
                STATE_NAMES[1]: STATE_COLORS[1],
                STATE_NAMES[2]: STATE_COLORS[2],
            },
            size="confidence",
            hover_data=["timestamp_formatted", "confidence"],
            labels={
                "timestamp": "Tiempo (segundos)",
                "predicted_state": "Estado Predicho",
                "state_name": "Estado",
            },
        )

        fig_timeline.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#cbd5e1",
            yaxis=dict(
                tickmode="array",
                tickvals=[0, 1, 2],
                ticktext=[STATE_NAMES[0], STATE_NAMES[1], STATE_NAMES[2]],
            ),
        )

        st.plotly_chart(fig_timeline, use_container_width=True)

        # Confianza a lo largo del tiempo
        st.markdown("### 📉 Confianza de Predicciones")

        fig_confidence = px.line(
            result.timeline_data,
            x="timestamp",
            y="confidence",
            color="state_name",
            color_discrete_map={
                STATE_NAMES[0]: STATE_COLORS[0],
                STATE_NAMES[1]: STATE_COLORS[1],
                STATE_NAMES[2]: STATE_COLORS[2],
            },
        )

        fig_confidence.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#cbd5e1",
            xaxis_title="Tiempo (segundos)",
            yaxis_title="Confianza",
        )

        st.plotly_chart(fig_confidence, use_container_width=True)

        # Tabla de datos crudos
        with st.expander("Ver datos completos del timeline"):
            st.dataframe(result.timeline_data, use_container_width=True)

# ============================================
# TAB 4: EXPORT
# ============================================
with tab_export:
    if "video_result" not in st.session_state:
        st.info("📤 Sube un video para exportar los resultados")
    else:
        result = st.session_state["video_result"]
        processor = st.session_state.get("processor")

        st.markdown("### 💾 Exportar Resultados")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 📄 Reporte de Texto")
            if processor:
                report_text = processor.generate_report(result)
                st.download_button(
                    label="Descargar Reporte (.txt)",
                    data=report_text,
                    file_name=f"reporte_{Path(result.video_path).stem}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

        with col2:
            st.markdown("#### 📊 Timeline CSV")
            csv_data = result.timeline_data.to_csv(index=False)
            st.download_button(
                label="Descargar Timeline (.csv)",
                data=csv_data,
                file_name=f"timeline_{Path(result.video_path).stem}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col3:
            st.markdown("#### 📑 Reporte Excel")
            try:
                import openpyxl

                excel_available = True
            except ImportError:
                excel_available = False
                st.warning(
                    "Instala 'openpyxl' para exportar a Excel: pip install openpyxl"
                )

            if excel_available and processor:
                excel_path = f"reporte_completo_{Path(result.video_path).stem}.xlsx"
                processor.export_to_excel(result, excel_path)
                with open(excel_path, "rb") as f:
                    st.download_button(
                        label="Descargar Excel (.xlsx)",
                        data=f.read(),
                        file_name=Path(excel_path).name,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )

# Footer
st.markdown("---")
st.markdown(
    """
<p style="text-align: center; color: #64748b;">
    🎬 Bonita White AI - Video Analytics 2026 | Computer Vision Team
</p>
""",
    unsafe_allow_html=True,
)
