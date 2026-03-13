"""
Test Script for Frame Extraction

Este script crea videos de prueba y ejecuta la extracción de frames
para verificar que todo funcione correctamente.
"""

import os
import sys
import numpy as np
import cv2
import tempfile
import shutil


def create_test_videos(
    output_dir: str, num_videos: int = 3, duration_sec: int = 3
) -> list:
    """
    Crea videos de prueba con colores sólidos.

    Args:
        output_dir: Directorio para guardar los videos
        num_videos: Número de videos a crear
        duration_sec: Duración en segundos de cada video

    Returns:
        Lista de rutas a los videos creados
    """
    print(f"📹 Creando {num_videos} videos de prueba...")

    os.makedirs(output_dir, exist_ok=True)

    videos = []
    classes = [
        ("prefloracion", (100, 200, 100)),  # Verde
        ("floracion_intermedia", (200, 200, 100)),  # Amarillo
        ("floracion_maxima", (200, 100, 100)),  # Rojo
    ]

    fps = 30
    frame_size = (640, 480)

    for i in range(num_videos):
        class_name, color = classes[i % len(classes)]
        filename = f"test_{class_name}_{i + 1:03d}.mp4"
        filepath = os.path.join(output_dir, filename)

        # Crear video
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(filepath, fourcc, fps, frame_size)

        for frame_idx in range(duration_sec * fps):
            # Crear frame con color sólido y número de frame
            frame = np.full((frame_size[1], frame_size[0], 3), color, dtype=np.uint8)

            # Añadir texto con número de frame
            text = f"Frame {frame_idx} - {class_name}"
            cv2.putText(
                frame, text, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
            )

            out.write(frame)

        out.release()
        videos.append(filepath)
        print(f"  ✅ Creado: {filename}")

    print(f"✅ Videos creados en: {output_dir}")
    return videos


def run_frame_extraction():
    """Ejecuta el script de extracción de frames."""
    print("\n🚀 Ejecutando extracción de frames...")

    # Importar el script de extracción
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from src.data.extract_frames import main as extract_main

    # Simular argumentos de línea de comandos
    sys.argv = [
        "extract_frames.py",
        "--video-dir",
        "bonita-white-classifier/data/raw/videos",
        "--output-dir",
        "bonita-white-classifier/data/processed/frames",
        "--workers",
        "2",
        "--clear-output",
        "--log-level",
        "INFO",
    ]

    try:
        extract_main()
        print("✅ Extracción de frames completada.")
        return True
    except Exception as e:
        print(f"❌ Error durante la extracción: {e}")
        import traceback

        traceback.print_exc()
        return False


def verify_output(output_dir: str) -> bool:
    """Verifica que los frames se hayan extraído correctamente."""
    print("\n🔍 Verificando salida...")

    if not os.path.exists(output_dir):
        print(f"❌ Directorio de salida no existe: {output_dir}")
        return False

    # Buscar clases
    classes = [
        "Estado_0_Prefloracion",
        "Estado_1_Floracion_Intermedia",
        "Estado_2_Floracion_Maxima",
    ]

    all_ok = True
    total_frames = 0

    for class_name in classes:
        class_dir = os.path.join(output_dir, class_name)

        if not os.path.exists(class_dir):
            print(f"⚠️  Directorio de clase no existe: {class_name}")
            continue

        frames = [f for f in os.listdir(class_dir) if f.endswith(".jpg")]
        total_frames += len(frames)

        if frames:
            print(f"  ✅ {class_name}: {len(frames)} frames")
        else:
            print(f"  ⚠️  {class_name}: sin frames")

    if total_frames > 0:
        print(f"\n✅ Total de frames extraídos: {total_frames}")
        return True
    else:
        print(f"\n❌ No se extrajeron frames.")
        return False


def cleanup(temp_dir: str):
    """Limpia directorios temporales."""
    print("\n🧹 Limpiando archivos temporales...")

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print(f"✅ Directorio temporal eliminado: {temp_dir}")


def main():
    """Función principal del script de prueba."""
    print("=" * 60)
    print("🧪 Test de Extracción de Frames")
    print("=" * 60)
    print()

    # Directorio temporal para videos de prueba
    temp_videos_dir = "bonita-white-classifier/data/raw/videos/test_videos"
    output_dir = "bonita-white-classifier/data/processed/frames"

    try:
        # Crear videos de prueba
        videos = create_test_videos(temp_videos_dir, num_videos=3, duration_sec=3)

        # Ejecutar extracción
        success = run_frame_extraction()

        if success:
            # Verificar salida
            verified = verify_output(output_dir)

            if verified:
                print("\n" + "=" * 60)
                print("🎉 Test completado exitosamente!")
                print("=" * 60)
                return 0
            else:
                print("\n" + "=" * 60)
                print("❌ Test fallido: No se verificó la salida")
                print("=" * 60)
                return 1
        else:
            print("\n" + "=" * 60)
            print("❌ Test fallido: Error en extracción")
            print("=" * 60)
            return 1

    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrumpido por el usuario.")
        return 130

    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback

        traceback.print_exc()
        return 1

    finally:
        # Opcional: limpiar archivos de prueba
        response = input("\n¿Desea eliminar los videos de prueba? (y/n): ").lower()
        if response == "y" or response == "yes":
            cleanup(temp_videos_dir)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
