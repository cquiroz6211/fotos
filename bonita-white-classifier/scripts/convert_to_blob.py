"""
Conversión ONNX -> .blob para OAK-1 (RVC2 / Myriad X) con normalización
ImageNet horneada en el modelo.

Por qué hornear la normalización:
- El modelo se entrenó con torchvision: x_norm = (x/255 - mean) / std
- Si feed-eamos uint8 [0,255] al .blob sin normalizar, las predicciones saturan
- OpenVINO Model Optimizer puede inyectar `(x - M) / S` al inicio del grafo:
    M = mean * 255 = (123.675, 116.28, 103.53)   # RGB
    S = std  * 255 = (58.395,  57.12,  57.375)   # RGB

El orden de canales asume que en runtime alimentamos RGB888p (lo cual hace
oak1_inference.py). Si llegaras a alimentar BGR, hay que invertir el orden o
agregar --reverse_input_channels.

Uso:
    python scripts/convert_to_blob.py \
        --input models/bonita_classifier_3clases.onnx \
        --output models/bonita_classifier_rvc2.blob

Requisitos:
    pip install blobconverter
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

# Forzar UTF-8 en stdout para Windows
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


# ImageNet stats x 255 (RGB)
IMAGENET_MEAN_255 = (123.675, 116.28, 103.53)
IMAGENET_SCALE_255 = (58.395, 57.12, 57.375)


def _inline_onnx_if_external(onnx_path: Path) -> Path:
    """
    Si el ONNX tiene weights externos (.onnx.data), genera una copia
    self-contained. blobconverter solo sube el .onnx, así que pesos externos
    rompen la compilación remota con FRAMEWORK ERROR.
    """
    import onnx

    has_external = any(p.suffix == ".data" and p.stem == onnx_path.name
                       for p in onnx_path.parent.iterdir())
    if not has_external and not onnx_path.with_suffix(".onnx.data").exists():
        return onnx_path

    inlined = onnx_path.with_name(onnx_path.stem + "_inlined.onnx")
    print(f"📦 ONNX con weights externos detectado, inlining -> {inlined.name}")
    model = onnx.load(str(onnx_path), load_external_data=True)
    onnx.save(model, str(inlined), save_as_external_data=False)
    return inlined


def convert(
    onnx_path: str,
    output_path: str,
    input_size: int = 224,
    shaves: int = 6,
):
    import blobconverter

    onnx_path = Path(onnx_path).resolve()
    output_path = Path(output_path).resolve()
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if not onnx_path.exists():
        print(f"❌ No existe: {onnx_path}")
        return False

    onnx_path = _inline_onnx_if_external(onnx_path)
    onnx_path_str = str(onnx_path)

    print(f"🔄 ONNX:  {onnx_path}")
    print(f"   .blob: {output_path}")
    print(f"   shaves: {shaves}  input: 1x3x{input_size}x{input_size} (RGB888p, U8)")
    print(f"   mean (x255): {IMAGENET_MEAN_255}")
    print(f"   scale (x255): {IMAGENET_SCALE_255}")
    print()

    mean_str = "[" + ",".join(map(str, IMAGENET_MEAN_255)) + "]"
    scale_str = "[" + ",".join(map(str, IMAGENET_SCALE_255)) + "]"

    # NO incluir --input_shape: el ONNX ya tiene batch=1 fijo (onnx-simplifier),
    # y pasarlo confunde el shape propagation de MO sobre el nodo Reshape del flatten.
    optimizer_params = [
        f"--mean_values={mean_str}",
        f"--scale_values={scale_str}",
        "--data_type=FP16",
    ]

    # -ip U8: le dice al compilador que el input es uint8 en runtime,
    # lo cual combina correctamente con mean/scale inyectados por MO.
    compile_params = ["-ip U8"]

    print("🔧 Compilando vía blobconverter (servidor Luxonis)…")
    blob_path = blobconverter.from_onnx(
        model=onnx_path_str,
        data_type="FP16",
        shaves=shaves,
        version="2021.4",
        optimizer_params=optimizer_params,
        compile_params=compile_params,
        use_cache=False,
        output_dir=str(output_dir),
    )

    blob_path = Path(blob_path)
    if blob_path != output_path:
        shutil.move(str(blob_path), str(output_path))

    print(f"✅ .blob listo: {output_path}")
    print(f"   tamaño: {output_path.stat().st_size / 1024:.1f} KB")
    return True


def main():
    parser = argparse.ArgumentParser(description="ONNX -> .blob para OAK-1 (RVC2)")
    parser.add_argument("--input", default="models/bonita_classifier_3clases.onnx")
    parser.add_argument("--output", default="models/bonita_classifier_rvc2.blob")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--shaves", type=int, default=6,
                        help="OAK-1 tiene 12 SHAVES; 6 es un buen default")
    args = parser.parse_args()

    ok = convert(args.input, args.output, args.input_size, args.shaves)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
