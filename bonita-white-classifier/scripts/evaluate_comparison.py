"""
Evaluacion comparativa real de los modelos entrenados.

Genera metricas reales:
- Accuracy
- Precision, Recall, F1 por clase
- Tiempo de inferencia
- Matriz de confusion
"""

import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.factory import create_model
from data.dataset import create_dataloaders


def evaluate_model_on_test(model, test_loader, device):
    """
    Evalua un modelo en el dataset de test.

    Returns:
        Dict con metricas y predicciones
    """
    model.eval()

    all_predictions = []
    all_labels = []
    all_probabilities = []

    inference_times = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Medir tiempo de inferencia
            start_time = time.time()
            outputs = model(images)
            end_time = time.time()

            batch_time = (end_time - start_time) / len(images)  # Tiempo por imagen
            inference_times.append(batch_time)

            # Obtener predicciones
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    # Calcular metricas
    accuracy = accuracy_score(all_labels, all_predictions)
    precision_macro = precision_score(
        all_labels, all_predictions, average="macro", zero_division=0
    )
    recall_macro = recall_score(
        all_labels, all_labels, average="macro", zero_division=0
    )
    f1_macro = f1_score(all_labels, all_predictions, average="macro", zero_division=0)

    # Metricas por clase
    precision_per_class = precision_score(
        all_labels, all_predictions, average=None, zero_division=0
    )
    recall_per_class = recall_score(
        all_labels, all_predictions, average=None, zero_division=0
    )
    f1_per_class = f1_score(all_labels, all_predictions, average=None, zero_division=0)

    # Matriz de confusion
    cm = confusion_matrix(all_labels, all_predictions)

    # Tiempo de inferencia
    avg_inference_time = np.mean(inference_times) * 1000  # en milisegundos

    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
        "f1_per_class": f1_per_class,
        "confusion_matrix": cm,
        "avg_inference_time_ms": avg_inference_time,
        "predictions": np.array(all_predictions),
        "labels": np.array(all_labels),
        "probabilities": np.array(all_probabilities),
    }


def load_trained_model(checkpoint_path, model_name, device):
    """Carga un modelo entrenado desde checkpoint."""

    # Crear modelo
    if model_name == "efficientnet_b0":
        model = create_model(
            model_name, num_classes=3, pretrained=False, device=str(device)
        )
    elif model_name == "mobilenet_v3_small":
        model = create_model(
            model_name, num_classes=3, pretrained=False, device=str(device)
        )
    else:
        raise ValueError(f"Modelo desconocido: {model_name}")

    # Cargar pesos
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    epoch = checkpoint.get("epoch", "unknown")
    metrics = checkpoint.get("metrics", {})

    return model, epoch, metrics


def print_comparison_table(results):
    """Imprime tabla comparativa de resultados."""

    print("\n" + "=" * 80)
    print("  COMPARACION REAL DE MODELOS - METRICAS EN DATASET DE TEST")
    print("=" * 80)

    # Tabla principal
    print("\n" + "-" * 80)
    print("  METRICAS GLOBALES")
    print("-" * 80)

    print(f"\n  {'Metrica':<25} {'EfficientNet-B0':>20} {'MobileNetV3-Small':>20}")
    print(f"  {'-' * 25} {'-' * 20} {'-' * 20}")

    for model_name, metrics in results.items():
        pass  # Solo necesitamos imprimir

    # Obtener nombres de modelos
    model_names = list(results.keys())

    print(
        f"  {'Accuracy':<25} {results[model_names[0]]['accuracy'] * 100:>19.2f}% {results[model_names[1]]['accuracy'] * 100:>19.2f}%"
    )
    print(
        f"  {'F1-Score (Macro)':<25} {results[model_names[0]]['f1_macro']:>20.4f} {results[model_names[1]]['f1_macro']:>20.4f}"
    )
    print(
        f"  {'Precision (Macro)':<25} {results[model_names[0]]['precision_macro']:>20.4f} {results[model_names[1]]['precision_macro']:>20.4f}"
    )
    print(
        f"  {'Recall (Macro)':<25} {results[model_names[0]]['recall_macro']:>20.4f} {results[model_names[1]]['recall_macro']:>20.4f}"
    )
    print(
        f"  {'Tiempo inferencia (ms)':<25} {results[model_names[0]]['avg_inference_time_ms']:>20.2f} {results[model_names[1]]['avg_inference_time_ms']:>20.2f}"
    )

    # Metricas por clase
    print("\n" + "-" * 80)
    print("  METRICAS POR CLASE")
    print("-" * 80)

    class_names = ["Estado_0_Prefloracion", "Estado_1_Intermedia", "Estado_2_Maxima"]

    for i, class_name in enumerate(class_names):
        print(f"\n  {class_name}:")
        print(f"    {'Metrica':<20} {'EfficientNet-B0':>15} {'MobileNetV3-Small':>15}")
        print(f"    {'-' * 20} {'-' * 15} {'-' * 15}")
        print(
            f"    {'Precision':<20} {results[model_names[0]]['precision_per_class'][i]:>15.4f} {results[model_names[1]]['precision_per_class'][i]:>15.4f}"
        )
        print(
            f"    {'Recall':<20} {results[model_names[0]]['recall_per_class'][i]:>15.4f} {results[model_names[1]]['recall_per_class'][i]:>15.4f}"
        )
        print(
            f"    {'F1-Score':<20} {results[model_names[0]]['f1_per_class'][i]:>15.4f} {results[model_names[1]]['f1_per_class'][i]:>15.4f}"
        )

    # Matrices de confusion
    print("\n" + "-" * 80)
    print("  MATRICES DE CONFUSION")
    print("-" * 80)

    for model_name, metrics in results.items():
        print(f"\n  {model_name.upper()}:")
        cm = metrics["confusion_matrix"]
        print(f"\n        Pred_0  Pred_1  Pred_2")
        print(f"  Real_0  {cm[0][0]:>6}  {cm[0][1]:>6}  {cm[0][2]:>6}")
        print(f"  Real_1  {cm[1][0]:>6}  {cm[1][1]:>6}  {cm[1][2]:>6}")
        print(f"  Real_2  {cm[2][0]:>6}  {cm[2][1]:>6}  {cm[2][2]:>6}")

    # Resumen final
    print("\n" + "-" * 80)
    print("  ANALISIS Y RECOMENDACION")
    print("-" * 80)

    effnet_acc = results["efficientnet_b0"]["accuracy"]
    mobilenet_acc = results["mobilenet_v3_small"]["accuracy"]
    acc_diff = (effnet_acc - mobilenet_acc) * 100

    effnet_time = results["efficientnet_b0"]["avg_inference_time_ms"]
    mobilenet_time = results["mobilenet_v3_small"]["avg_inference_time_ms"]
    speedup = effnet_time / mobilenet_time if mobilenet_time > 0 else 0

    print(f"""
  DIFERENCIA DE PRECISION:
    - EfficientNet es {abs(acc_diff):.2f}% {"mejor" if acc_diff > 0 else "peor"}
    - MobileNetV3-Small alcanza {mobilenet_acc * 100:.2f}% de accuracy
  
  DIFERENCIA DE VELOCIDAD:
    - MobileNetV3-Small es {speedup:.1f}x mas rapido
    - EfficientNet tarda {effnet_time:.2f}ms por imagen
    - MobileNetV3-Small tarda {mobilenet_time:.2f}ms por imagen
  
  RECOMENDACION PARA OAK-1:
    {"MobileNetV3-Small" if mobilenet_acc > 0.8 else "EfficientNet-B0"} es la mejor opcion
    - Precision: {mobilenet_acc * 100:.2f}% (suficiente para clasificacion)
    - Velocidad: {mobilenet_time:.2f}ms ({1000 / mobilenet_time:.0f} FPS teoricos)
    - Tamano: 14.9 MB vs 56.5 MB
    """)

    print("=" * 80)


def main():
    """Funcion principal."""

    print("\n" + "=" * 80)
    print("  EVALUACION COMPARATIVA DE MODELOS ENTRENADOS")
    print("=" * 80)

    # Setup
    device = torch.device("cpu")  # Usar CPU para comparacion justa
    print(f"\n  Dispositivo: {device}")

    # Cargar dataloader de test
    print("\n  Cargando dataset de test...")
    dataloaders = create_dataloaders(
        data_dir="data/splits",
        batch_size=32,
        num_workers=0,
        img_size=224,
        pin_memory=False,
    )
    test_loader = dataloaders.get("test")

    if test_loader is None:
        print("  ERROR: No se encontro dataset de test")
        return

    print(f"  Dataset de test: {len(test_loader.dataset)} imagenes")

    # Modelos a evaluar
    models_to_evaluate = {
        "efficientnet_b0": "checkpoints/best_model.pth",
        "mobilenet_v3_small": "checkpoints/mobilenet_small/best_model.pth",
    }

    results = {}

    # Evaluar cada modelo
    for model_name, checkpoint_path in models_to_evaluate.items():
        if not os.path.exists(checkpoint_path):
            print(
                f"\n  [SKIP] {model_name} - Checkpoint no encontrado: {checkpoint_path}"
            )
            continue

        print(f"\n  Evaluando {model_name}...")
        print(f"  Checkpoint: {checkpoint_path}")

        # Cargar modelo
        model, epoch, train_metrics = load_trained_model(
            checkpoint_path, model_name, device
        )
        print(f"  Epoca de entrenamiento: {epoch}")

        # Evaluar
        metrics = evaluate_model_on_test(model, test_loader, device)
        results[model_name] = metrics

        print(f"  Accuracy: {metrics['accuracy'] * 100:.2f}%")
        print(f"  F1-Score: {metrics['f1_macro']:.4f}")
        print(f"  Tiempo inferencia: {metrics['avg_inference_time_ms']:.2f}ms")

    # Mostrar comparacion
    if len(results) >= 2:
        print_comparison_table(results)
    else:
        print("\n  ERROR: Se necesitan al menos 2 modelos para comparar")

    return results


if __name__ == "__main__":
    results = main()
