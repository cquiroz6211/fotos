# Guia Rapida: Estructura del Proyecto y Comparacion de Modelos

## Estructura del Proyecto

```
bonita-white-classifier/
│
├── src/                           ← CODIGO FUENTE
│   ├── models/                     ← Modelos de IA
│   │   ├── factory.py              ← Fabrica de modelos (crea cualquier modelo)
│   │   ├── efficientnet/           ← EfficientNet-B0
│   │   │   ├── __init__.py
│   │   │   └── model.py
│   │   └── mobilenet/              ← MobileNetV3 (Small + Large)
│   │       ├── __init__.py
│   │       └── model.py
│   │
│   ├── training/                   ← Entrenamiento
│   │   ├── train.py                ← Script principal de entrenamiento
│   │   └── evaluate.py             ← Evaluacion de modelos
│   │
│   ├── data/                       ← Manejo de datos
│   │   ├── dataset.py              ← Carga de imagenes
│   │   └── split_dataset.py        ← Division train/val/test
│   │
│   ├── utils/                      ← UTILIDADES (nuevo)
│   │   ├── logging_utils.py        ← Configuracion de logs
│   │   ├── config_utils.py         ← Carga de configs YAML
│   │   ├── device_utils.py         ← Seleccion de CPU/GPU
│   │   ├── training_utils.py       ← EarlyStopping, class weights
│   │   └── metrics.py              ← Metricas (accuracy, F1, etc.)
│   │
│   └── inference/                  ← Inferencia/Deployment
│       └── video_processor.py
│
├── configs/                        ← CONFIGURACIONES
│   ├── efficientnet/
│   │   └── efficientnet_b0.yaml
│   └── mobilenet/
│       ├── mobilenet_v3_small.yaml
│       └── mobilenet_v3_large.yaml
│
├── scripts/                        ← SCRIPTS UTILES
│   ├── compare_models.py           ← Compara los 3 modelos
│   └── model_comparison.py         ← Reporte detallado (nuevo)
│
├── checkpoints/                    ← MODELOS ENTRENADOS
│   ├── best_model.pth              ← EfficientNet-B0 entrenado
│   └── mobilenet_small/            ← MobileNetV3-Small entrenado
│       ├── best_model.pth
│       └── final_model.pth
│
├── data/                           ← DATOS
│   ├── raw/videos/                 ← Videos originales
│   ├── processed/frames/           ← Frames extraidos
│   └── splits/                     ← Dataset dividido
│       ├── train/                  ← 530 imagenes
│       ├── val/                    ← 115 imagenes
│       └── test/                   ← 116 imagenes
│
├── logs/                           ← LOGS DE ENTRENAMIENTO
│   └── tensorboard/                ← Visualizacion con TensorBoard
│
└── DOCUMENTACION_MODELOS.md        ← Documentacion completa
```

---

## Comparacion de los 3 Modelos

### Tabla Resumen

| Modelo | Parametros | MACs | FPS (OAK-1) | Estado |
|--------|-----------|------|-------------|--------|
| **EfficientNet-B0** | 4.7M | 390M | 8-15 FPS | Entrenado |
| **MobileNetV3-Small** | 1.2M | 56M | 40-60 FPS | Entrenado |
| **MobileNetV3-Large** | 3.5M | 218M | 15-25 FPS | No entrenado |

### ¿Cual elegir para OAK-1?

```
┌─────────────────────────────────────────────────────────────┐
│                    RECOMENDACION FINAL                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Para OAK-1: MobileNetV3-Small                              │
│                                                             │
│  ¿Por que?                                                  │
│  - 7x MAS RAPIDO que EfficientNet (56M vs 390M MACs)       │
│  - 4x MENOS PARAMETROS (1.2M vs 4.7M)                      │
│  - TAMANO: 14.2 MB vs 53.9 MB                              │
│  - FPS: 40-60 (tiempo real suave) vs 8-15 (lento)          │
│                                                             │
│  Trade-off:                                                 │
│  - Puede tener ~2-3% menos precision                       │
│  - Pero para 3 clases de flores, es suficiente             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Comandos Utiles

### Ver comparacion de modelos
```bash
cd bonita-white-classifier
python scripts/model_comparison.py
```

### Entrenar un modelo
```bash
# Entrenar MobileNetV3-Small
python src/training/train.py --model mobilenet_v3_small

# Entrenar MobileNetV3-Large (no entrenado aun)
python src/training/train.py --model mobilenet_v3_large

# Entrenar EfficientNet-B0
python src/training/train.py --model efficientnet_b0
```

### Evaluar un modelo
```bash
# Evaluar en test
python src/training/evaluate.py --checkpoint checkpoints/mobilenet_small/best_model.pth

# Evaluar en validacion
python src/training/evaluate.py --checkpoint checkpoints/best_model.pth --split val
```

### Ver logs de entrenamiento
```bash
# Con TensorBoard
tensorboard --logdir logs/tensorboard

# Ver archivo de log
cat logs/train_*.log
```

---

## Diferencias Arquitectonicas

### MobileNetV3-Small vs MobileNetV3-Large

| Caracteristica | Small | Large |
|---------------|-------|-------|
| Capas bottleneck | 11 | 17 |
| Features finales | 512 | 1280 |
| Dropout | 0.2 | 0.2 |
| Activacion | Hard-Swish | Hard-Swish |

**Impacto:** Large tiene 3x mas features, lo que le da mas capacidad de aprendizaje pero es 4x mas lento.

### MobileNetV3 vs EfficientNet-B0

| Caracteristica | MobileNetV3 | EfficientNet-B0 |
|---------------|-------------|-----------------|
| Bloques | Bottleneck | MBConv |
| Atencion | SE opcional | SE en cada bloque |
| Activacion | Hard-Swish | Swish |
| Objetivo | Edge/Mobile | Precision |

**Impacto:** Hard-Swish es mas eficiente en hardware edge (OAK-1).

---

## Modelos Entrenados Actuales

### EfficientNet-B0
- Archivo: `checkpoints/best_model.pth`
- Tamano: 53.9 MB
- Epocas: 5+ (ver logs para detalle exacto)

### MobileNetV3-Small
- Archivo: `checkpoints/mobilenet_small/best_model.pth`
- Tamano: 14.2 MB
- Epocas: 10+

### MobileNetV3-Large
- Estado: **No entrenado**
- Para entrenar: `python src/training/train.py --model mobilenet_v3_large`

---

## Proximos Pasos

1. **Entrenar MobileNetV3-Large** para tener comparacion completa
2. **Evaluar los 3 modelos** en el dataset de test
3. **Exportar a ONNX** para deployment en OAK-1
4. **Cuantizar a INT8** para optimizar tamano y velocidad

---

## Contacto

Para dudas sobre el proyecto, revisar:
- `DOCUMENTACION_MODELOS.md` - Documentacion tecnica completa
- `scripts/model_comparison.py` - Script de comparacion
- Logs en `logs/` - Historial de entrenamiento