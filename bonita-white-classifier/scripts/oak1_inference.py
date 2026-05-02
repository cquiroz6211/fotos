"""
Inference en OAK-1 para clasificación Bonita White

Este script ejecuta inferencia en tiempo real usando la cámara OAK-1
y el modelo .blob convertido para clasificar el estado fenológico.

Uso:
    python scripts/oak1_inference.py --model models/bonita_classifier.blob

Clases:
    0: Prefloración
    1: Floración Intermedia  
    2: Floración Máxima
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np

# Forzar UTF-8 en stdout para consolas Windows (cp1252 no soporta emojis)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# Agregar src al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Importar depthai
import depthai as dai


# Configuración de clases
CLASS_NAMES = {
    0: "Prefloración",
    1: "Floración Intermedia",
    2: "Floración Máxima"
}

CLASS_COLORS = {
    0: (0, 255, 0),    # Verde
    1: (255, 255, 0),  # Amarillo
    2: (255, 0, 0)     # Rojo
}


def build_pipeline(model_path: str, input_size: int = 224):
    """
    Crea el pipeline DepthAI v3 y devuelve (pipeline, q_rgb, q_nn).

    En v3 las salidas se consumen vía createOutputQueue() directamente,
    sin XLinkOut. El pipeline luego se arranca con pipeline.start().
    """
    pipeline = dai.Pipeline()

    # Cámara - en v3 Camera reemplaza ColorCamera
    cam = pipeline.create(dai.node.Camera).build()
    # RGB planar (CHW) — el modelo fue entrenado con torchvision (RGB) y el NN
    # espera planar. Si mandamos BGR888i (HWC) sale el warning de layout.
    cam_output = cam.requestOutput(
        (input_size, input_size),
        dai.ImgFrame.Type.RGB888p,
        fps=30,
    )

    # Red neuronal
    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(model_path)
    nn.setNumInferenceThreads(2)

    # Conexión: cámara -> NN
    cam_output.link(nn.input)

    # Output queues (v3 puro: sin XLinkOut)
    q_rgb = cam_output.createOutputQueue(maxSize=4, blocking=False)
    q_nn = nn.out.createOutputQueue(maxSize=4, blocking=False)

    return pipeline, q_rgb, q_nn


def preprocess_frame(frame, input_size=224):
    """
    Preprocesa el frame para el modelo.
    
    Args:
        frame: Frame de la cámara
        input_size: Tamaño de entrada
    
    Returns:
        Tensor preprocesado
    """
    import cv2
    
    # Redimensionar
    resized = cv2.resize(frame, (input_size, input_size))
    
    # Normalizar con valores ImageNet
    # Convertir a float32 y normalizar a [0, 1]
    normalized = resized.astype(np.float32) / 255.0
    
    # Aplicar normalización ImageNet
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    normalized = (normalized - mean) / std
    
    # Transponer de HWC a CHW
    transposed = np.transpose(normalized, (2, 0, 1))
    
    return transposed


def softmax(x):
    """Aplica softmax a las predicciones."""
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()


def postprocess_output(output_data):
    """
    Procesa la salida del modelo.
    
    Args:
        output_data: Salida cruda del modelo
    
    Returns:
        Diccionario con clase predicha y probabilidades
    """
    # La salida puede venir en diferentes formatos
    if isinstance(output_data, list):
        output = np.array(output_data).flatten()
    else:
        output = output_data.flatten()
    
    # Aplicar softmax para obtener probabilidades
    probabilities = softmax(output)
    
    # Obtener clase predicha
    predicted_class = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_class])
    
    return {
        'class': predicted_class,
        'class_name': CLASS_NAMES[predicted_class],
        'confidence': confidence,
        'probabilities': probabilities.tolist()
    }


def run_inference(
    model_path: str,
    input_size: int = 224,
    show_preview: bool = True,
    confidence_threshold: float = 0.5
):
    """
    Ejecuta la inferencia en tiempo real.
    
    Args:
        model_path: Ruta al modelo .blob
        input_size: Tamaño de entrada
        show_preview: Mostrar preview de video
        confidence_threshold: Umbral de confianza mínimo
    """
    print("🚀 Iniciando OAK-1 Inference...")
    print(f"   Modelo: {model_path}")
    print(f"   Input size: {input_size}")
    print(f"   Preview: {show_preview}")
    print()
    
    # Verificar que el modelo existe
    if not Path(model_path).exists():
        print(f"❌ Error: No se encontró el modelo en {model_path}")
        return
    
    # Crear pipeline + queues (v3)
    print("📷 Conectando a OAK-1...")
    pipeline, q_rgb, q_nn = build_pipeline(model_path, input_size)
    pipeline.start()

    print("✅ OAK-1 conectada y lista!")
    print("   Presiona 'q' para salir")
    print()
    
    # Variables para FPS
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    try:
        import cv2
        
        while True:
            # Obtener frame de la cámara
            in_rgb = q_rgb.tryGet()
            
            if in_rgb is not None:
                frame = in_rgb.getCvFrame()
                
                # Preprocesar para inference local (opcional)
                # En OAK-1, la inferencia se hace en el dispositivo
                
                frame_count += 1
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
            
            # Obtener resultado de inferencia
            in_nn = q_nn.tryGet()
            
            if in_nn is not None:
                # Procesar salida
                output_data = in_nn.getFirstTensor()
                result = postprocess_output(output_data)
                
                # Mostrar resultados en consola
                print(f"🎯 Predicción: {result['class_name']} "
                      f"(confianza: {result['confidence']:.2%}) | "
                      f"FPS: {fps:.1f}")
            
            # Mostrar preview si está habilitado
            if show_preview and in_rgb is not None:
                frame = in_rgb.getCvFrame()
                
                # Dibujar información en el frame
                if in_nn is not None:
                    result = postprocess_output(in_nn.getFirstTensor())
                    
                    # Color según clase
                    color = CLASS_COLORS[result['class']]
                    
                    # Texto de resultado
                    label = f"{result['class_name']}: {result['confidence']:.1%}"
                    cv2.putText(frame, label, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    
                    # Barra de confianza
                    bar_width = int(200 * result['confidence'])
                    cv2.rectangle(frame, (10, 50), 
                                 (10 + bar_width, 70), color, -1)
                
                # FPS
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow("OAK-1 - Bonita White Classifier", frame)
            
            # Salir con 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Interrumpido por el usuario")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Limpiar
        try:
            pipeline.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("✅ Conexión cerrada")


def run_inference_simple(model_path: str, num_frames: int = 10):
    """
    Modo simple: ejecuta inference en N frames y muestra resultados.
    
    Args:
        model_path: Ruta al modelo .blob
        num_frames: Número de frames a procesar
    """
    print(f"🔍 Ejecutando {num_frames} predicciones de prueba...")
    
    if not Path(model_path).exists():
        print(f"❌ Error: No se encontró el modelo en {model_path}")
        return
    
    # Crear pipeline (v3)
    pipeline, q_rgb, q_nn = build_pipeline(model_path)
    pipeline.start()

    results = []

    try:
        for i in range(num_frames):
            in_nn = q_nn.get()
            _ = q_rgb.get()  # drenar la cola de video para que no se llene

            output_data = in_nn.getFirstTensor()
            result = postprocess_output(output_data)

            results.append(result)

            print(f"  Frame {i+1}: {result['class_name']} ({result['confidence']:.2%})")

            time.sleep(0.1)
    finally:
        pipeline.stop()
    
    # Resumen
    print("\n📊 Resumen de predicciones:")
    for class_id, class_name in CLASS_NAMES.items():
        count = sum(1 for r in results if r['class'] == class_id)
        print(f"   {class_name}: {count}/{num_frames}")


def main():
    parser = argparse.ArgumentParser(
        description='Inference en OAK-1 para Bonita White Classifier'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='models/bonita_classifier_rvc2.blob',
        help='Ruta al modelo .blob'
    )
    parser.add_argument(
        '--input-size',
        type=int,
        default=224,
        help='Tamaño de entrada del modelo'
    )
    parser.add_argument(
        '--no-preview',
        action='store_true',
        help='No mostrar preview de video'
    )
    parser.add_argument(
        '--test-frames',
        type=int,
        default=0,
        help='Modo test: ejecutar N frames y salir'
    )
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.5,
        help='Umbral de confianza mínimo'
    )
    
    args = parser.parse_args()
    
    if args.test_frames > 0:
        run_inference_simple(args.model, args.test_frames)
    else:
        run_inference(
            model_path=args.model,
            input_size=args.input_size,
            show_preview=not args.no_preview,
            confidence_threshold=args.confidence_threshold
        )


if __name__ == '__main__':
    main()