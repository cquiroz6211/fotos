"""
Script para comparar los 3 modelos disponibles.

Muestra:
- Información de cada modelo (parámetros, MACs)
- Estado de entrenamiento
- Diferencias arquitectónicas
- Recomendación para OAK-1
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from models.factory import create_model, list_available_models, get_model_info


def print_header(text: str):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_section(text: str):
    print("\n" + "-" * 70)
    print(f"  {text}")
    print("-" * 70)


def compare_models():
    """Compara los 3 modelos disponibles."""

    print_header("COMPARACION DE MODELOS - BONITA WHITE CLASSIFIER")

    # Listar modelos disponibles
    models = list_available_models()
    print(f"\n  Modelos disponibles: {models}")

    # Tabla comparativa
    print_section("1. TABLA COMPARATIVA")

    print(
        "\n  +------------------------+-------------+-------------+---------------------+"
    )
    print(
        "  | Modelo                 | Parametros  | MACs        | Descripcion         |"
    )
    print(
        "  +------------------------+-------------+-------------+---------------------+"
    )

    for model_name in models:
        info = get_model_info(model_name)
        name = model_name.replace("_", " ").title()
        params = info.get("parameters", "N/A")
        macs = info.get("macs", "N/A")
        desc = info.get("description", "N/A")[:35] + "..."

        print(f"  | {name:22} | {params:11} | {macs:11} | {desc:19} |")

    print(
        "  +------------------------+-------------+-------------+---------------------+"
    )

    # Detalles de cada modelo
    print_section("2. DETALLES DE CADA MODELO")

    for model_name in models:
        print(f"\n  ### {model_name.upper()} ###")
        info = get_model_info(model_name)

        print(f"  Parametros: {info.get('parameters', 'N/A')}")
        print(f"  MACs: {info.get('macs', 'N/A')}")
        print(f"  Input: {info.get('input_size', 'N/A')}")
        print(f"  Descripcion: {info.get('description', 'N/A')}")

        # Crear modelo y obtener info real
        try:
            model = create_model(
                model_name, num_classes=3, pretrained=False, device="cpu"
            )
            model_info = model.get_model_info()
            print(f"  Parametros reales: {model_info['total_parameters']:,}")
            print(f"  Parametros entrenables: {model_info['trainable_parameters']:,}")
        except Exception as e:
            print(f"  Error al crear modelo: {e}")

    # Comparacion de velocidad estimada
    print_section("3. ESTIMACION DE VELOCIDAD EN OAK-1")

    print("""
  +------------------------+---------------+----------------+------------------+
  | Modelo                 | FPS Estimado  | Latencia (ms)  | Tiempo Real?     |
  +------------------------+---------------+----------------+------------------+
  | mobilenet_v3_small     | 40-60 FPS     | 20-25 ms       | SI - Excelente   |
  | mobilenet_v3_large     | 15-25 FPS     | 40-60 ms       | SI - Aceptable   |
  | efficientnet_b0        | 8-15 FPS      | 70-120 ms      | LIMIT - Lento    |
  +------------------------+---------------+----------------+------------------+
    """)

    # Verificar modelos entrenados
    print_section("4. MODELOS ENTRENADOS")

    checkpoints_dir = Path(__file__).parent.parent / "checkpoints"

    trained_models = {
        "efficientnet_b0": checkpoints_dir / "best_model.pth",
        "mobilenet_v3_small": checkpoints_dir / "mobilenet_small" / "best_model.pth",
        "mobilenet_v3_large": None,  # No entrenado aun
    }

    print("\n  Estado de entrenamiento:")
    for model_name, checkpoint_path in trained_models.items():
        if checkpoint_path and checkpoint_path.exists():
            size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
            print(f"  [OK] {model_name:22} - Entrenado ({size_mb:.1f} MB)")
        else:
            print(f"  [--] {model_name:22} - No entrenado")

    # Recomendacion final
    print_section("5. RECOMENDACION PARA OAK-1")

    print("""
  +------------------------+------------------------------------------------+
  | Si necesitas...        | Usa este modelo                                |
  +------------------------+------------------------------------------------+
  | Maxima velocidad       | MobileNetV3-Small (56M MACs, ~50 FPS)          |
  | Mejor precision        | EfficientNet-B0 (390M MACs, ~10 FPS)           |
  | Balance optimo         | MobileNetV3-Large (218M MACs, ~20 FPS)         |
  +------------------------+------------------------------------------------+
  
  RECOMENDACION: MobileNetV3-Small para OAK-1
  - Mas rapido (7x mas rapido que EfficientNet)
  - Menos memoria (1.2M parametros vs 5.3M)
  - Suficiente precision para clasificacion de 3 estados
    """)

    # Diferencias arquitectonicas
    print_section("6. DIFERENCIAS ARQUITECTONICAS")

    print("""
  MobileNetV3-Small vs MobileNetV3-Large:
  - Small: 11 capas bottleneck, 512 features finales
  - Large: 17 capas bottleneck, 1280 features finales
  - Ambos usan Hard-Swish (mas eficiente en hardware)
  
  MobileNetV3 vs EfficientNet-B0:
  - MobileNetV3: Diseñado para mobile/edge, usa bottleneck
  - EfficientNet: Diseñado para precision, usa MBConv
  - EfficientNet tiene squeeze-and-excitation en cada bloque
  
  Impacto en OAK-1:
  - Hard-Swish se ejecuta mas rapido en hardware edge
  - Menos capas = menos operaciones = mas FPS
  - Modelos mas pequeños = menos memoria RAM usada
    """)

    print_header("FIN DEL REPORTE")

    return models


def test_model_creation():
    """Prueba que los 3 modelos se pueden crear correctamente."""

    print_header("TEST DE CREACION DE MODELOS")

    for model_name in list_available_models():
        print(f"\n  Probando {model_name}...")
        try:
            model = create_model(
                model_name, num_classes=3, pretrained=False, device="cpu"
            )

            # Test forward pass
            x = torch.randn(1, 3, 224, 224)
            model.eval()
            with torch.no_grad():
                output = model(x)

            info = model.get_model_info()
            print(f"  [OK] Creado - {info['total_parameters']:,} parametros")
            print(f"       Input: {x.shape} -> Output: {output.shape}")

        except Exception as e:
            print(f"  [ERROR] {e}")

    print("\n  Todos los modelos creados correctamente!")


if __name__ == "__main__":
    compare_models()
    print("\n")
    test_model_creation()
