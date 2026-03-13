"""
Quick Setup Script for Frame Extraction

Este script ayuda a configurar rápidamente el entorno para extracción de frames.
"""

import os
import subprocess
import sys


def install_dependencies():
    """Instala las dependencias necesarias."""
    print("📦 Instalando dependencias...")

    dependencies = [
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ]

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + dependencies)
        print("✅ Dependencias instaladas correctamente.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando dependencias: {e}")
        sys.exit(1)


def create_directory_structure():
    """Crea la estructura de directorios necesaria."""
    print("📁 Creando estructura de directorios...")

    directories = [
        "bonita-white-classifier/data/raw/videos/dia1-4",
        "bonita-white-classifier/data/raw/videos/dia5-8",
        "bonita-white-classifier/data/raw/videos/dia9-11",
        "bonita-white-classifier/data/processed/frames",
        "bonita-white-classifier/logs",
        "bonita-white-classifier/config",
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✅ {directory}")

    print("✅ Estructura de directorios creada.")


def verify_installation():
    """Verifica que las dependencias estén instaladas correctamente."""
    print("🔍 Verificando instalación...")

    packages = {
        "cv2": "OpenCV",
        "numpy": "NumPy",
        "yaml": "PyYAML",
        "tqdm": "tqdm",
    }

    all_installed = True

    for module, name in packages.items():
        try:
            if module == "yaml":
                import yaml
            else:
                __import__(module)
            print(f"  ✅ {name} está instalado")
        except ImportError:
            print(f"  ❌ {name} NO está instalado")
            all_installed = False

    return all_installed


def print_next_steps():
    """Imprime los siguientes pasos para el usuario."""
    print("\n" + "=" * 60)
    print("🎉 Configuración completada con éxito!")
    print("=" * 60)
    print("\n📝 Siguientes pasos:")
    print(
        "1. Coloque sus videos en: bonita-white-classifier/data/raw/videos/{dia1-4,dia5-8,dia9-11}/"
    )
    print(
        "2. Configure el mapeo de clases en: bonita-white-classifier/config/extract_frames_config.yaml"
    )
    print("3. Ejecute la extracción:")
    print("   python bonita-white-classifier/src/data/extract_frames.py")
    print("\n💡 Para más opciones:")
    print("   python bonita-white-classifier/src/data/extract_frames.py --help")
    print("=" * 60)


def main():
    """Función principal del script de setup."""
    print("=" * 60)
    print("⚙️  Bonita White Classifier - Frame Extraction Setup")
    print("=" * 60)
    print()

    # Preguntar si instalar dependencias
    response = input("¿Desea instalar las dependencias necesarias? (y/n): ").lower()

    if response == "y" or response == "yes":
        install_dependencies()
    else:
        print("⚠️  Saltando instalación de dependencias.")

    print()

    # Crear estructura de directorios
    create_directory_structure()

    print()

    # Verificar instalación
    if verify_installation():
        print("\n✅ Todas las dependencias están instaladas correctamente.")
    else:
        print("\n⚠️  Algunas dependencias faltan. Ejecute:")
        print("   pip install opencv-python numpy pyyaml tqdm")
        sys.exit(1)

    print()

    # Imprimir siguientes pasos
    print_next_steps()


if __name__ == "__main__":
    main()
