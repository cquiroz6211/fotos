# Despliegue en OAK-1 — Bonita White Classifier

Guía completa para compilar el modelo y correrlo en tiempo real con la cámara OAK-1.

---

## Arquitectura del pipeline

```
OAK-1 (RGB888p 224x224)
  └─► NeuralNetwork (.blob, normalización ImageNet baked-in)
        └─► softmax en host → clase + confianza
```

La normalización ImageNet se hornea al momento de compilar el `.blob`:

```
x_norm = (x_uint8 - mean*255) / (std*255)
mean = [0.485, 0.456, 0.406]  →  [123.675, 116.28, 103.53]
std  = [0.229, 0.224, 0.225]  →  [58.395,  57.12,  57.375]
```

Esto permite alimentar directamente bytes uint8 de la cámara sin preprocesamiento en el host.

---

## Archivos

```
models/
├── bonita_classifier_rvc2.blob        ← modelo en producción para OAK-1
├── bonita_classifier_sim.onnx         ← fuente del blob (self-contained, opset 11+)
└── bonita_classifier_rvc2.no_norm.blob.bak  ← backup del blob sin normalización

scripts/
├── oak1_inference.py                  ← inference en tiempo real (DepthAI v3)
└── convert_to_blob.py                 ← conversión ONNX → .blob
```

---

## 1. Compilar el modelo (.blob)

Sólo necesario cuando cambia el ONNX. El blob ya está compilado en `models/`.

```bash
python scripts/convert_to_blob.py \
    --input  models/bonita_classifier_sim.onnx \
    --output models/bonita_classifier_rvc2.blob \
    --shaves 6
```

**Notas de compilación:**
- Usa `bonita_classifier_sim.onnx` (self-contained). Los otros `.onnx` tienen weights externos (`.onnx.data`) y blobconverter no los acepta.
- No pasar `--input_shape` al Model Optimizer: el ONNX ya tiene batch=1 fijo, y pasarlo confunde la inferencia de shapes en el nodo `node_view` (Reshape flatten del EfficientNet).
- `compile_params=["-ip U8"]` es obligatorio para que el compilador acepte uint8 de la cámara junto con los mean/scale inyectados por MO.

---

## 2. Correr inference en la OAK-1

### Requisito previo

Conectar la OAK-1 por USB-C. Verificar que el dispositivo es detectado:

```bash
python -c "import depthai as dai; [print(d) for d in dai.Device.getAllAvailableDevices()]"
```

### Modo interactivo (con preview de video)

```bash
python scripts/oak1_inference.py --model models/bonita_classifier_rvc2.blob
```

Abre una ventana con:
- Preview de la cámara
- Predicción + confianza superpuesta
- FPS en tiempo real

Presionar `q` para salir.

### Modo test (sin ventana, N frames)

```bash
python scripts/oak1_inference.py --model models/bonita_classifier_rvc2.blob --test-frames 10
```

### Todas las opciones

| Opción | Default | Descripción |
|--------|---------|-------------|
| `--model` | `models/bonita_classifier_rvc2.blob` | Ruta al .blob |
| `--input-size` | `224` | Resolución de entrada |
| `--no-preview` | `False` | Desactivar ventana de video |
| `--test-frames N` | `0` (infinito) | Correr N frames y salir |
| `--confidence-threshold` | `0.5` | Umbral mínimo de confianza |

---

## 3. Clases

| ID | Clase | Color en preview |
|----|-------|-----------------|
| 0 | Prefloración | Verde |
| 1 | Floración Intermedia | Amarillo |
| 2 | Floración Máxima | Rojo |

---

## 4. Versiones y compatibilidad

| Componente | Versión |
|-----------|---------|
| depthai | 3.5.0 |
| DepthAI API | v3 (sin XLinkOut — usa `createOutputQueue()`) |
| OpenVINO (compilación) | 2021.4 |
| Dispositivo | OAK-1, RVC2 / Myriad X |
| Input del blob | RGB888p (planar), uint8, 1×3×224×224 |
| Shaves | 6 de 12 disponibles |

---

## 5. Solución de problemas

### `No devices found`
- Verificar cable USB-C
- `python -c "import depthai as dai; print(dai.Device.getAllAvailableDevices())"`

### Predicciones siempre con 100% de confianza
- El `.blob` fue compilado sin normalización ImageNet. Recompilar con `convert_to_blob.py`.

### `AttributeError: module 'depthai.node' has no attribute 'XLinkOut'`
- API depthai v3 no tiene `XLinkOut`. Usar `createOutputQueue()` directamente sobre los outputs del pipeline (ya corregido en `oak1_inference.py`).

### Warning: `Input image is interleaved (HWC), NN specifies planar (CHW)`
- La cámara debe configurarse con `dai.ImgFrame.Type.RGB888p` (planar), no `BGR888i`. Ya corregido.

### `HTTPError 400` en blobconverter con ONNX externo
- Los archivos `bonita_classifier_3clases.onnx` y `bonita_classifier_legacy.onnx` tienen weights en archivo separado (`.onnx.data`). blobconverter no los acepta. Usar siempre `bonita_classifier_sim.onnx`.

### `Reshape node_view mismatch` en OpenVINO MO
- No pasar `--input_shape` en `optimizer_params`. El script ya lo excluye por defecto.
