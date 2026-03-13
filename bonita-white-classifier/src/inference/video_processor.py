"""
Video Processor Module for Bonita White AI

Este modulo procesa videos completos, analizando frame por frame
y generando reportes agregados del estado fenologico del cultivo.

Autor: Computer Vision Team
Fecha: 2026
"""

import os
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Generator
from dataclasses import dataclass
from collections import Counter
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class FramePrediction:
    """Resultado de prediccion para un frame individual."""

    frame_number: int
    timestamp: float  # segundos desde inicio del video
    predicted_class: int
    confidence: float
    probabilities: np.ndarray


@dataclass
class VideoAnalysisResult:
    """Resultado completo del analisis de un video."""

    video_path: str
    total_frames: int
    processed_frames: int
    fps: float
    duration_seconds: float
    predictions: List[FramePrediction]

    # Agregados
    dominant_state: int
    state_distribution: Dict[int, float]
    confidence_mean: float
    confidence_std: float
    state_transitions: List[Tuple[float, int, int]]  # tiempo, de_estado, a_estado

    # Timeline
    timeline_data: pd.DataFrame


class VideoProcessor:
    """
    Procesador de videos para clasificacion fenologica.

    Analiza videos frame por frame y genera reportes agregados
    del estado del cultivo a lo largo del tiempo.

    Args:
        model: Modelo PyTorch entrenado
        device: Dispositivo para inferencia ('cpu', 'cuda', etc.)
        frame_interval: Procesar 1 frame cada N (default: 30 = 1 frame/segundo a 30fps)
        confidence_threshold: Umbral minimo de confianza para considerar prediccion valida
    """

    # Nombres de estados para reportes
    STATE_NAMES = {0: "Prefloracion", 1: "Floracion Intermedia", 2: "Floracion Maxima"}

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cpu",
        frame_interval: int = 30,
        confidence_threshold: float = 0.5,
    ):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        self.frame_interval = frame_interval
        self.confidence_threshold = confidence_threshold

        logger.info(
            f"VideoProcessor inicializado: device={device}, "
            f"frame_interval={frame_interval}"
        )

    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocesa un frame para el modelo.

        Args:
            frame: Frame BGR de OpenCV

        Returns:
            Tensor normalizado listo para inferencia
        """
        # Convertir BGR a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize a 224x224
        frame_resized = cv2.resize(frame_rgb, (224, 224))

        # Normalizacion ImageNet
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        frame_normalized = frame_resized / 255.0
        frame_normalized = (frame_normalized - mean) / std

        # Convertir a tensor: HWC -> CHW
        tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1).float()

        return tensor.unsqueeze(0).to(self.device)

    def predict_frame(self, frame: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """
        Predice la clase de un frame individual.

        Returns:
            (clase_predicha, confianza, probabilidades)
        """
        tensor = self.preprocess_frame(frame)

        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            probs = probabilities[0].cpu().numpy()

        return predicted_class, confidence, probs

    def process_video(
        self, video_path: str, progress_callback: Optional[callable] = None
    ) -> VideoAnalysisResult:
        """
        Procesa un video completo y genera reporte de analisis.

        Args:
            video_path: Ruta al archivo de video
            progress_callback: Funcion callback(progress_pct, current_frame, total_frames)

        Returns:
            VideoAnalysisResult con todas las metricas agregadas
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video no encontrado: {video_path}")

        # Abrir video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video_path}")

        # Obtener propiedades del video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration_seconds = total_frames / fps if fps > 0 else 0

        logger.info(f"Procesando video: {video_path.name}")
        logger.info(
            f"  Frames: {total_frames}, FPS: {fps:.2f}, "
            f"Duracion: {duration_seconds:.2f}s"
        )

        # Procesar frames
        predictions = []
        frame_count = 0
        processed_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Procesar solo 1 frame cada N
            if frame_count % self.frame_interval == 0:
                timestamp = frame_count / fps if fps > 0 else 0

                try:
                    pred_class, confidence, probs = self.predict_frame(frame)

                    prediction = FramePrediction(
                        frame_number=frame_count,
                        timestamp=timestamp,
                        predicted_class=pred_class,
                        confidence=confidence,
                        probabilities=probs,
                    )
                    predictions.append(prediction)
                    processed_count += 1

                except Exception as e:
                    logger.warning(f"Error procesando frame {frame_count}: {e}")

            frame_count += 1

            # Callback de progreso
            if progress_callback and frame_count % 10 == 0:
                progress_pct = (frame_count / total_frames) * 100
                progress_callback(progress_pct, frame_count, total_frames)

        cap.release()

        logger.info(f"Video procesado: {processed_count} frames analizados")

        # Calcular agregados
        result = self._calculate_aggregates(
            video_path=str(video_path),
            total_frames=total_frames,
            processed_frames=processed_count,
            fps=fps,
            duration_seconds=duration_seconds,
            predictions=predictions,
        )

        return result

    def _calculate_aggregates(
        self,
        video_path: str,
        total_frames: int,
        processed_frames: int,
        fps: float,
        duration_seconds: float,
        predictions: List[FramePrediction],
    ) -> VideoAnalysisResult:
        """Calcula metricas agregadas del analisis."""

        if not predictions:
            raise ValueError("No se pudieron procesar frames del video")

        # Distribucion de estados
        state_counts = Counter([p.predicted_class for p in predictions])
        total_preds = len(predictions)

        state_distribution = {
            state: (count / total_preds) * 100 for state, count in state_counts.items()
        }

        # Estado predominante
        dominant_state = state_counts.most_common(1)[0][0]

        # Estadisticas de confianza
        confidences = [p.confidence for p in predictions]
        confidence_mean = np.mean(confidences)
        confidence_std = np.std(confidences)

        # Detectar transiciones de estado
        state_transitions = []
        for i in range(1, len(predictions)):
            prev_pred = predictions[i - 1]
            curr_pred = predictions[i]

            if prev_pred.predicted_class != curr_pred.predicted_class:
                state_transitions.append(
                    (
                        curr_pred.timestamp,
                        prev_pred.predicted_class,
                        curr_pred.predicted_class,
                    )
                )

        # Crear timeline DataFrame
        timeline_data = pd.DataFrame(
            [
                {
                    "timestamp": p.timestamp,
                    "timestamp_formatted": str(timedelta(seconds=int(p.timestamp))),
                    "frame": p.frame_number,
                    "predicted_state": p.predicted_class,
                    "state_name": self.STATE_NAMES[p.predicted_class],
                    "confidence": p.confidence,
                    "prob_state_0": p.probabilities[0],
                    "prob_state_1": p.probabilities[1],
                    "prob_state_2": p.probabilities[2],
                }
                for p in predictions
            ]
        )

        return VideoAnalysisResult(
            video_path=video_path,
            total_frames=total_frames,
            processed_frames=processed_frames,
            fps=fps,
            duration_seconds=duration_seconds,
            predictions=predictions,
            dominant_state=dominant_state,
            state_distribution=state_distribution,
            confidence_mean=confidence_mean,
            confidence_std=confidence_std,
            state_transitions=state_transitions,
            timeline_data=timeline_data,
        )

    def generate_report(self, result: VideoAnalysisResult) -> str:
        """
        Genera un reporte en texto del analisis.

        Returns:
            String con el reporte formateado
        """
        report = []
        report.append("=" * 60)
        report.append("REPORTE DE ANALISIS DE VIDEO - Bonita White AI")
        report.append("=" * 60)
        report.append("")
        report.append(f"Video: {Path(result.video_path).name}")
        report.append(
            f"Fecha de analisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        report.append("")
        report.append("-" * 60)
        report.append("INFORMACION DEL VIDEO")
        report.append("-" * 60)
        report.append(f"Duracion: {result.duration_seconds:.2f} segundos")
        report.append(f"Total frames: {result.total_frames}")
        report.append(f"Frames analizados: {result.processed_frames}")
        report.append(f"FPS del video: {result.fps:.2f}")
        report.append("")
        report.append("-" * 60)
        report.append("RESULTADOS DEL ANALISIS")
        report.append("-" * 60)
        report.append(f"Estado predominante: {self.STATE_NAMES[result.dominant_state]}")
        report.append("")
        report.append("Distribucion de estados:")
        for state, pct in sorted(result.state_distribution.items()):
            report.append(f"  {self.STATE_NAMES[state]}: {pct:.1f}%")
        report.append("")
        report.append(f"Confianza promedio: {result.confidence_mean:.2%}")
        report.append(f"Desviacion estandar: {result.confidence_std:.2%}")
        report.append("")

        if result.state_transitions:
            report.append("-" * 60)
            report.append("TRANSICIONES DETECTADAS")
            report.append("-" * 60)
            for timestamp, from_state, to_state in result.state_transitions:
                time_str = str(timedelta(seconds=int(timestamp)))
                report.append(
                    f"  {time_str}: {self.STATE_NAMES[from_state]} -> "
                    f"{self.STATE_NAMES[to_state]}"
                )
            report.append("")

        report.append("=" * 60)

        return "\n".join(report)

    def export_to_csv(self, result: VideoAnalysisResult, output_path: str):
        """Exporta el timeline a CSV."""
        result.timeline_data.to_csv(output_path, index=False)
        logger.info(f"Timeline exportado a: {output_path}")

    def export_to_excel(self, result: VideoAnalysisResult, output_path: str):
        """Exporta reporte completo a Excel con multiples hojas."""
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            # Hoja 1: Timeline detallado
            result.timeline_data.to_excel(writer, sheet_name="Timeline", index=False)

            # Hoja 2: Resumen
            summary_data = {
                "Metrica": [
                    "Video",
                    "Duracion (s)",
                    "Frames analizados",
                    "Estado predominante",
                    "Confianza promedio",
                ],
                "Valor": [
                    Path(result.video_path).name,
                    result.duration_seconds,
                    result.processed_frames,
                    self.STATE_NAMES[result.dominant_state],
                    f"{result.confidence_mean:.2%}",
                ],
            }
            pd.DataFrame(summary_data).to_excel(
                writer, sheet_name="Resumen", index=False
            )

            # Hoja 3: Distribucion
            dist_data = {
                "Estado": [
                    self.STATE_NAMES[s] for s in result.state_distribution.keys()
                ],
                "Porcentaje": list(result.state_distribution.values()),
            }
            pd.DataFrame(dist_data).to_excel(
                writer, sheet_name="Distribucion", index=False
            )

        logger.info(f"Reporte exportado a Excel: {output_path}")


def create_video_processor(
    checkpoint_path: str = "checkpoints/best_model.pth",
    device: str = "cpu",
    frame_interval: int = 30,
) -> VideoProcessor:
    """
    Factory function para crear un VideoProcessor con modelo cargado.

    Args:
        checkpoint_path: Ruta al checkpoint del modelo
        device: Dispositivo para inferencia
        frame_interval: Intervalo entre frames a procesar

    Returns:
        VideoProcessor configurado y listo para usar
    """
    from models.efficientnet_model import EfficientNetClassifier

    logger.info(f"Cargando modelo desde: {checkpoint_path}")
    model, info = EfficientNetClassifier.load_checkpoint(checkpoint_path, device=device)

    return VideoProcessor(model=model, device=device, frame_interval=frame_interval)
