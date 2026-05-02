"""
Conversión de modelo PyTorch a ONNX para OAK-1

Este script convierte el modelo EfficientNet-B0 entrenado al formato ONNX,
que es requerido para la conversión posterior a .blob para la cámara OAK-1.

Uso:
    python scripts/convert_to_onnx.py --checkpoint checkpoints/best_model.pth --output models/bonita_classifier.onnx
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
from pathlib import Path

# Agregar src al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.efficientnet_model import EfficientNetClassifier


def convert_pytorch_to_onnx(
    checkpoint_path: str,
    output_path: str,
    num_classes: int = 3,
    input_size: int = 224,
    opset_version: int = 12
):
    """
    Convierte un modelo PyTorch a formato ONNX.
    
    Args:
        checkpoint_path: Ruta al archivo .pth del modelo
        output_path: Ruta donde se guardará el modelo ONNX
        num_classes: Número de clases del modelo
        input_size: Tamaño de entrada esperado (default: 224)
        opset_version: Versión de opset de ONNX (default: 12)
    """
    print(f"🔄 Cargando modelo desde: {checkpoint_path}")
    
    # Crear instancia del modelo
    model = EfficientNetClassifier(num_classes=num_classes, pretrained=False)
    
    # Cargar pesos del checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Manejar diferentes formatos de checkpoint
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    print("✅ Modelo cargado correctamente")
    
    # Crear tensor de entrada dummy
    # ONNX requiere un batch size fijo o dinámico
    dummy_input = torch.randn(1, 3, input_size, input_size)
    
    # Asegurar que el directorio de salida existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"🔄 Convirtiendo a ONNX (opset version: {opset_version})...")
    
    # Exportar a ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✅ Modelo ONNX guardado en: {output_path}")
    
    # Verificar el modelo
    verify_onnx_model(output_path, dummy_input)
    
    return output_path


def verify_onnx_model(onnx_path: str, dummy_input: torch.Tensor):
    """Verifica que el modelo ONNX sea válido."""
    import onnx
    
    print("🔍 Verificando modelo ONNX...")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    # Verificar inferencia
    import numpy as np
    import onnxruntime as ort
    
    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name
    
    dummy_np = dummy_input.numpy()
    output = ort_session.run(None, {input_name: dummy_np})
    
    print(f"✅ Verificación exitosa. Output shape: {output[0].shape}")
    print(f"   Probabilidades: {output[0][0]}")


def main():
    parser = argparse.ArgumentParser(
        description='Convertir modelo PyTorch a ONNX para OAK-1'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/best_model.pth',
        help='Ruta al checkpoint del modelo'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models/bonita_classifier.onnx',
        help='Ruta de salida para el modelo ONNX'
    )
    parser.add_argument(
        '--num-classes',
        type=int,
        default=3,
        help='Número de clases'
    )
    parser.add_argument(
        '--input-size',
        type=int,
        default=224,
        help='Tamaño de entrada del modelo'
    )
    parser.add_argument(
        '--opset',
        type=int,
        default=12,
        help='Versión de opset de ONNX'
    )
    
    args = parser.parse_args()
    
    convert_pytorch_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        num_classes=args.num_classes,
        input_size=args.input_size,
        opset_version=args.opset
    )


if __name__ == '__main__':
    main()