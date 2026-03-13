"""
Script para comparar múltiples modelos de clasificación.

Entrena los 3 modelos y genera comparación de métricas:
- EfficientNet-B0
- MobileNetV3-Small
- MobileNetV3-Large
"""

import os
import sys
import subprocess
import json
import logging
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.factory import list_available_models, get_model_info

logger = logging.getLogger(__name__)


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def train_model(model_name: str, config_path: str) -> dict:
    """
    Train a model and return results.

    Args:
        model_name: Name of the model
        config_path: Path to config file

    Returns:
        Dictionary with training results
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Training model: {model_name}")
    logger.info(f"Config: {config_path}")
    logger.info(f"{'=' * 60}\n")

    # Run training
    cmd = [
        sys.executable,
        "-m",
        "src.training.train",
        "--model",
        model_name,
        "--config",
        config_path,
    ]

    logger.info(f"Running: {' '.join(cmd)}")

    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(Path(__file__).parent.parent)
    )

    if result.returncode != 0:
        logger.error(f"Training failed for {model_name}")
        logger.error(result.stderr)
        return {"status": "failed", "error": result.stderr}

    logger.info(f"Training completed for {model_name}")
    logger.info(result.stdout)

    return {"status": "success", "output": result.stdout}


def compare_models(results: dict) -> str:
    """
    Generate comparison report.

    Args:
        results: Dictionary with results for each model

    Returns:
        Markdown formatted comparison report
    """
    report = []
    report.append("# Comparación de Modelos - Bonita White Classification")
    report.append(f"\nFecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\n{'=' * 80}\n")

    # Table header
    report.append("## Resumen de Modelos\n")
    report.append("| Modelo | Parámetros | MACs | Estado |")
    report.append("|--------|-----------|------|--------|")

    for model_name, result in results.items():
        info = get_model_info(model_name)
        status = "✅ Éxito" if result.get("status") == "success" else "❌ Falló"
        report.append(
            f"| {model_name} | {info.get('parameters', 'N/A')} | "
            f"{info.get('macs', 'N/A')} | {status} |"
        )

    report.append("\n## Detalles\n")

    for model_name, result in results.items():
        report.append(f"\n### {model_name}\n")
        info = get_model_info(model_name)
        report.append(f"**Descripción:** {info.get('description', 'N/A')}")
        report.append(f"\n**Parámetros:** {info.get('parameters', 'N/A')}")
        report.append(f"\n**MACs:** {info.get('macs', 'N/A')}")
        report.append(f"\n**Estado:** {result.get('status', 'N/A')}")

        if result.get("status") == "failed":
            report.append(f"\n**Error:** {result.get('error', 'Unknown')}")

    report.append(f"\n{'=' * 80}\n")
    report.append("\n## Notas para OAK-1\n")
    report.append(
        "- **MobileNetV3-Small:** ~56M MACs - Ideal para OAK-1, menor latencia"
    )
    report.append(
        "- **MobileNetV3-Large:** ~218M MACs - Buen balance precisión/velocidad"
    )
    report.append("- **EfficientNet-B0:** ~390M MACs - Mayor precisión, más pesado")

    return "\n".join(report)


def main():
    """Main comparison function."""
    setup_logging()

    logger.info("Starting model comparison experiment")
    logger.info("Available models: " + ", ".join(list_available_models()))

    # Define model configs
    model_configs = {
        "efficientnet_b0": "configs/efficientnet_b0.yaml",
        "mobilenet_v3_small": "configs/mobilenet/mobilenet_v3_small.yaml",
        "mobilenet_v3_large": "configs/mobilenet/mobilenet_v3_large.yaml",
    }

    # Train each model
    results = {}
    for model_name, config_path in model_configs.items():
        if not os.path.exists(config_path):
            logger.warning(f"Config not found: {config_path}")
            continue

        results[model_name] = train_model(model_name, config_path)

    # Generate comparison report
    report = compare_models(results)

    # Save report
    output_dir = Path("experiments/comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"comparison_report_{timestamp}.md"

    with open(report_path, "w") as f:
        f.write(report)

    logger.info(f"\nComparison report saved to: {report_path}")
    logger.info("\n" + report)


if __name__ == "__main__":
    main()
