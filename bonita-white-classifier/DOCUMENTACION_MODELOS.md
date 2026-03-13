# Documentación Técnica: Comparación de Modelos para Clasificación Fenológica

## Proyecto: Bonita White - Clasificador de Estado Fenológico

**Fecha:** 13 de Marzo, 2026  
**Autor:** Equipo de IA - Visión Computacional  
**Versión:** 1.0

---

## Tabla de Contenidos

1. [Resumen Ejecutivo](#1-resumen-ejecutivo)
2. [Contexto del Problema](#2-contexto-del-problema)
3. [Modelos Evaluados](#3-modelos-evaluados)
4. [Arquitectura de Cada Modelo](#4-arquitectura-de-cada-modelo)
5. [Dataset y Preprocesamiento](#5-dataset-y-preprocesamiento)
6. [Configuración de Entrenamiento](#6-configuración-de-entrenamiento)
7. [Comparativa Técnica](#7-comparativa-técnica)
8. [Flujo de Inferencia](#8-flujo-de-inferencia)
9. [Recomendaciones para OAK-1](#9-recomendaciones-para-oak-1)
10. [Cómo Usar Este Sistema](#10-cómo-usar-este-sistema)

---

## 1. Resumen Ejecutivo

### Objetivo del Proyecto

Desarrollar un sistema de visión por computadora capaz de clasificar el **estado fenológico** del cultivo de flores **Bonita White** en 3 categorías:

| Clase | Estado | Días | Descripción Visual |
|-------|--------|------|-------------------|
| **0** | Prefloración | 1-4 | Campo verde, pocas flores visibles |
| **1** | Floración Intermedia | 5-8 | 40-60% cobertura blanca |
| **2** | Floración Máxima | 9-11 | 80-90% blanco, listo para corte |

### Restricción Principal

El modelo debe ejecutarse en una **cámara OAK-1** (dispositivo edge con Intel Myriad X), lo que impone limitaciones estrictas de:
- **Tamaño del modelo:** < 10 MB preferiblemente
- **Operaciones (MACs):** < 300-500M para inferencia en tiempo real
- **Latencia:** < 100ms por frame para video fluido

### Modelos Evaluados

Se evaluaron **3 arquitecturas** de redes neuronales convolucionales:

1. **EfficientNet-B0** - Balance entre precisión y eficiencia
2. **MobileNetV3-Small** - Optimizado para dispositivos edge (más rápido)
3. **MobileNetV3-Large** - Más preciso que Small, aún eficiente

---

## 2. Contexto del Problema

### ¿Por Qué Este Problema?

En la agricultura de precisión, especialmente en cultivos de flores para exportación como el **Gypsophila paniculata** (conocido comercialmente como "Bonita White" o "Gypsophila"), el momento óptimo de cosecha es crítico:

- **Cosechar muy temprano:** Flores no han alcanzado tamaño/calidad óptima → menor valor comercial
- **Cosechar muy tarde:** Flores pierden frescura, pétalos se dañan → menor vida útil

### Solución Tradicional vs. Visión Computacional

| Método | Ventaja | Desventaja |
|--------|---------|------------|
| **Inspección visual humana** | Precisa, considera contexto | Lenta, subjetiva, costosa |
| **Visión computacional** | Rápida, objetiva, escalable | Requiere datos, infraestructura |

### El Reto Técnico

Desplegar un modelo de deep learning en un **dispositivo edge** (OAK-1) que:
- Tiene potencia de procesamiento limitada (~1 TOPS)
- No tiene acceso a GPU dedicada
- Debe procesar video en tiempo real (~24-30 FPS)
- Opera en condiciones de campo (sin conexión cloud constante)

---

## 3. Modelos Evaluados

### 3.1 EfficientNet-B0

**Descripción:** Arquitectura de red neuronal convolucional que utiliza **compound scaling** para balancear profundidad, ancho y resolución.

| Característica | Valor |
|---------------|-------|
| **Parámetros totales** | ~5.3 millones |
| **Operaciones (MACs)** | ~390 millones |
| **Tamaño del modelo** | ~20 MB (FP32) |
| **Input recomendado** | 224×224 píxeles |
| **Año** | 2019 (Google) |

**Por qué se eligió:**
- Excelente balance precisión/eficiencia en benchmarks
- Ampliamente usado en producción
- Transfer learning robusto desde ImageNet

**Contra para OAK-1:**
- 390M MACs está en el límite superior para tiempo real
- Puede requerir cuantización INT8 para deployment

---

### 3.2 MobileNetV3-Small

**Descripción:** Versión "pequeña" de MobileNetV3, optimizada específicamente para dispositivos móviles y edge con recursos limitados.

| Característica | Valor |
|---------------|-------|
| **Parámetros totales** | ~2.5 millones |
| **Operaciones (MACs)** | ~56 millones |
| **Tamaño del modelo** | ~10 MB (FP32) |
| **Input recomendado** | 224×224 píxeles |
| **Año** | 2019 (Google) |

**Por qué se eligió:**
- **El más ligero de los 3 modelos**
- 7× menos operaciones que EfficientNet-B0
- Diseñado específicamente para edge/mobile
- Usa **Hard-Swish** como activación (más eficiente en hardware)

**Contra:**
- Menor precisión potencial que los otros modelos
- Puede no capturar features complejas de las flores

---

### 3.3 MobileNetV3-Large

**Descripción:** Versión "grande" de MobileNetV3, mantiene la eficiencia arquitectónica pero con más capacidad de representación.

| Característica | Valor |
|---------------|-------|
| **Parámetros totales** | ~5.5 millones |
| **Operaciones (MACs)** | ~218 millones |
| **Tamaño del modelo** | ~21 MB (FP32) |
| **Input recomendado** | 224×224 píxeles |
| **Año** | 2019 (Google) |

**Por qué se eligió:**
- **Punto medio ideal** entre Small y EfficientNet
- ~2× menos operaciones que EfficientNet-B0
- Misma familia arquitectónica que Small (fácil comparación)
- Buen balance para deployment en edge con algo más de potencia

**Contra:**
- Más pesado que Small
- Todavía requiere optimización para OAK-1

---

## 4. Arquitectura de Cada Modelo

### 4.1 EfficientNet-B0

```
┌─────────────────────────────────────────────────────────────┐
│                    EfficientNet-B0                          │
├─────────────────────────────────────────────────────────────┤
│  Input: 224×224×3                                           │
│    ↓                                                        │
│  Conv2D (3×3, stride=2) + BatchNorm + Swish                 │
│    ↓                                                        │
│  MBConv1 (×1) - Mobile Inverted Bottleneck                  │
│    ↓                                                        │
│  MBConv6 (×2)                                               │
│    ↓                                                        │
│  MBConv6 (×2)                                               │
│    ↓                                                        │
│  MBConv6 (×3)                                               │
│    ↓                                                        │
│  MBConv6 (×3)                                               │
│    ↓                                                        │
│  MBConv6 (×1)                                               │
│    ↓                                                        │
│  Global Average Pooling                                     │
│    ↓                                                        │
│  Dropout (p=0.3)                                            │
│    ↓                                                        │
│  Dense (512) + ReLU + BatchNorm                             │
│    ↓                                                        │
│  Dropout (p=0.15)                                           │
│    ↓                                                        │
│  Dense (3) - Output (3 clases)                              │
└─────────────────────────────────────────────────────────────┘
```

**Características clave:**
- **MBConv blocks:** Convoluciones depthwise separables + squeeze-and-excitation
- **Swish activation:** Función de activación no monótona
- **Depthwise separable convolutions:** Reduce parámetros vs convoluciones estándar

---

### 4.2 MobileNetV3-Small

```
┌─────────────────────────────────────────────────────────────┐
│                   MobileNetV3-Small                         │
├─────────────────────────────────────────────────────────────┤
│  Input: 224×224×3                                           │
│    ↓                                                        │
│  Conv2D (3×3, stride=2) + BatchNorm + HardSwish             │
│    ↓                                                        │
│  SE-Bottleneck (×1) - Squeeze & Excitation                  │
│    ↓                                                        │
│  Bottleneck (×2)                                            │
│    ↓                                                        │
│  SE-Bottleneck (×2)                                         │
│    ↓                                                        │
│  Bottleneck (×2)                                            │
│    ↓                                                        │
│  SE-Bottleneck (×1)                                         │
│    ↓                                                        │
│  Conv2D (1×1) + BatchNorm + HardSwish                       │
│    ↓                                                        │
│  Global Average Pooling                                     │
│    ↓                                                        │
│  Dense (512) + HardSwish                                    │
│    ↓                                                        │
│  Dropout (p=0.2)                                            │
│    ↓                                                        │
│  Dense (3) - Output (3 clases)                              │
└─────────────────────────────────────────────────────────────┘
```

**Características clave:**
- **Hard-Swish activation:** Aproximación polinómica de Swish (más rápida en hardware)
- **Squeeze-and-Excitation (SE):** Atención por canal para mejorar representación
- **Bottleneck blocks:** Reduce dimensiones antes de convolución

---

### 4.3 MobileNetV3-Large

```
┌─────────────────────────────────────────────────────────────┐
│                   MobileNetV3-Large                         │
├─────────────────────────────────────────────────────────────┤
│  Input: 224×224×3                                           │
│    ↓                                                        │
│  Conv2D (3×3, stride=2) + BatchNorm + HardSwish             │
│    ↓                                                        │
│  Bottleneck (×1)                                            │
│    ↓                                                        │
│  SE-Bottleneck (×1)                                         │
│    ↓                                                        │
│  Bottleneck (×3)                                            │
│    ↓                                                        │
│  SE-Bottleneck (×3)                                         │
│    ↓                                                        │
│  SE-Bottleneck (×3)                                         │
│    ↓                                                        │
│  Bottleneck (×2)                                            │
│    ↓                                                        │
│  Bottleneck (×2)                                            │
│    ↓                                                        │
│  SE-Bottleneck (×2)                                         │
│    ↓                                                        │
│  Conv2D (1×1) + BatchNorm + HardSwish                       │
│    ↓                                                        │
│  Global Average Pooling                                     │
│    ↓                                                        │
│  Dense (1280) + HardSwish                                   │
│    ↓                                                        │
│  Dropout (p=0.2)                                            │
│    ↓                                                        │
│  Dense (3) - Output (3 clases)                              │
└─────────────────────────────────────────────────────────────┘
```

**Características clave:**
- Más bloques que Small (mayor profundidad)
- Feature map de 1280 dimensiones antes del clasificador (vs 512 en Small)
- Mismas optimizaciones de hardware que Small

---

## 5. Dataset y Preprocesamiento

### 5.1 Composición del Dataset

| Split | Cantidad | Porcentaje |
|-------|----------|------------|
| **Entrenamiento** | 530 imágenes | 69.6% |
| **Validación** | 115 imágenes | 15.1% |
| **Prueba** | 116 imágenes | 15.3% |
| **TOTAL** | **761 imágenes** | **100%** |

### 5.2 Distribución por Clase

| Clase | Entrenamiento | Validación | Prueba | Total |
|-------|--------------|------------|--------|-------|
| **Estado 0 (Prefloración)** | 219 | 48 | 48 | 315 |
| **Estado 1 (Intermedia)** | 177 | 38 | 39 | 254 |
| **Estado 2 (Máxima)** | 134 | 29 | 29 | 192 |

**Nota:** El dataset está **ligeramente desbalanceado** hacia Estado 0. Se utilizan **class weights** durante el entrenamiento para compensar.

### 5.3 Preprocesamiento de Imágenes

**Transformaciones aplicadas:**

```python
# Para ENTRENAMIENTO (con data augmentation):
transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),           # Redimensionar a 224×224
    transforms.RandomHorizontalFlip(p=0.5),  # Volteo horizontal aleatorio
    transforms.RandomRotation(degrees=15),   # Rotación aleatoria ±15°
    transforms.ColorJitter(                  # Variación de color
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    transforms.RandomAffine(                 # Transformación afín
        degrees=0,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1)
    ),
    transforms.ToTensor(),                   # Convertir a tensor
    transforms.Normalize(                    # Normalización ImageNet
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    transforms.RandomErasing(p=0.3),         # Oclusión aleatoria
])

# Para VALIDACIÓN/PRUEBA (sin augmentation):
transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])
```

### 5.4 ¿Por Qué Esta Normalización?

Los valores de normalización `[0.485, 0.456, 0.406]` (mean) y `[0.229, 0.224, 0.225]` (std) son las **estadísticas de ImageNet**.

**Razón:** Todos los modelos (EfficientNet, MobileNetV3) fueron **pre-entrenados en ImageNet**. Usar las mismas estadísticas asegura que la distribución de input sea consistente con lo que el modelo "espera" ver.

---

## 6. Configuración de Entrenamiento

### 6.1 Hiperparámetros Comunes

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| **Función de pérdida** | CrossEntropyLoss | Estándar para clasificación multiclase |
| **Optimizador** | AdamW | Adam con decaimiento de peso corregido |
| **Learning rate** | 0.001 | Valor estándar para fine-tuning |
| **Weight decay** | 0.0001 | Regularización L2 para evitar overfitting |
| **Batch size** | 16-32 | Depende del modelo (ver abajo) |
| **Épocas máximas** | 50 | Suficiente para convergencia |
| **Early stopping** | 10 épocas | Detiene si no hay mejora en validación |
| **Scheduler** | CosineAnnealingLR | Reduce LR suavemente durante entrenamiento |
| **Warmup** | 3 épocas | Calentamiento gradual del learning rate |

### 6.2 Configuración Específica por Modelo

| Parámetro | EfficientNet-B0 | MobileNetV3-Small | MobileNetV3-Large |
|-----------|-----------------|-------------------|-------------------|
| **Batch size** | 16 | 32 | 24 |
| **Dropout rate** | 0.3 | 0.2 | 0.25 |
| **Freeze base** | False | False | False |
| **Class weights** | Sí | Sí | Sí |

**¿Por qué diferentes batch sizes?**
- **MobileNetV3-Small** es más pequeño → cabe más en memoria → batch más grande (32)
- **EfficientNet-B0** es más pesado → batch más pequeño (16)
- Batch size más grande = entrenamiento más estable y rápido

### 6.3 Transfer Learning Strategy

**Enfoque utilizado: Fine-tuning completo**

```
┌─────────────────────────────────────────────────────────────┐
│  Estrategia de Transfer Learning                            │
├─────────────────────────────────────────────────────────────┤
│  1. Cargar pesos pre-entrenados en ImageNet                 │
│  2. Reemplazar capa clasificadora final                     │
│     (originalmente 1000 clases → ahora 3 clases)            │
│  3. Entrenar TODAS las capas (no congelar backbone)         │
│  4. Usar learning rate pequeño (0.001) para ajuste fino     │
└─────────────────────────────────────────────────────────────┘
```

**¿Por qué no congelar el backbone?**
- Dataset pequeño (761 imágenes) sugiere congelar para evitar overfitting
- **PERO** las 3 clases son muy específicas (flores blancas en diferentes estados)
- Las features de ImageNet pueden no capturar las diferencias sutiles
- **Decisión:** Fine-tuning completo con regularización (dropout, early stopping)

### 6.4 Class Weights

Para manejar el desbalance de clases, se calculan pesos inversamente proporcionales a la frecuencia:

```python
# Fórmula: weight[class] = total_samples / (num_classes * count[class])

Estado 0: 0.7619  # Más frecuente → peso menor
Estado 1: 1.0667  # Frecuencia media
Estado 2: 1.3333  # Menos frecuente → peso mayor
```

**Efecto:** El modelo "presta más atención" a la clase menos frecuente durante el entrenamiento.

---

## 7. Comparativa Técnica

### 7.1 Tabla Comparativa Completa

| Característica | EfficientNet-B0 | MobileNetV3-Small | MobileNetV3-Large |
|---------------|-----------------|-------------------|-------------------|
| **Parámetros** | 5.3M | 2.5M | 5.5M |
| **Parámetros reales (entrenables)** | 4.7M | 1.2M | 3.5M |
| **MACs (operaciones)** | 390M | 56M | 218M |
| **Tamaño modelo (FP32)** | ~20 MB | ~10 MB | ~21 MB |
| **Tamaño modelo (INT8)** | ~5 MB | ~2.5 MB | ~5 MB |
| **Activación** | Swish | Hard-Swish | Hard-Swish |
| **Feature extractor** | MBConv | Bottleneck + SE | Bottleneck + SE |
| **Dimensiones finales** | 512 | 512 | 1280 |
| **Batch size usado** | 16 | 32 | 24 |
| **Dropout** | 0.3 | 0.2 | 0.25 |

### 7.2 Comparación de Complejidad

```
┌─────────────────────────────────────────────────────────────┐
│  Complejidad Relativa (referencia: MobileNetV3-Small = 1x)  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  MobileNetV3-Small:   1.0x (base)                           │
│  MobileNetV3-Large:   3.9x más operaciones                  │
│  EfficientNet-B0:     7.0x más operaciones                  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  MobileNetV3-Small   ████████ (56M MACs)             │  │
│  │  MobileNetV3-Large   ████████████████████████████ (218M)│  │
│  │  EfficientNet-B0     ████████████████████████████████████│  │
│  │                    ██████████████████████████ (390M)  │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 7.3 Estimación de Rendimiento en OAK-1

| Modelo | FPS Estimado (OAK-1) | Latencia Estimada | Viable para Tiempo Real? |
|--------|---------------------|-------------------|-------------------------|
| **MobileNetV3-Small** | ~40-60 FPS | ~20-25 ms | ✅ Sí, excelente |
| **MobileNetV3-Large** | ~15-25 FPS | ~40-60 ms | ✅ Sí, aceptable |
| **EfficientNet-B0** | ~8-15 FPS | ~70-120 ms | ⚠️ Límite, puede funcionar |

**Notas:**
- Estimaciones basadas en benchmarks públicos de OAK-1 con modelos similares
- FPS reales dependen de resolución de input y optimización del modelo
- **Recomendación:** MobileNetV3-Small para máxima velocidad, Large para mejor precisión

### 7.4 Análisis de Memoria

| Modelo | RAM para Inferencia | RAM para Entrenamiento |
|--------|---------------------|------------------------|
| **MobileNetV3-Small** | ~50 MB | ~500 MB |
| **MobileNetV3-Large** | ~100 MB | ~1 GB |
| **EfficientNet-B0** | ~150 MB | ~1.5 GB |

**OAK-1 tiene 512 MB RAM** → Todos los modelos caben para inferencia.

---

## 8. Flujo de Inferencia

### 8.1 Flujo General (Los 3 Modelos)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FLUJO DE INFERENCIA                         │
└─────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐
  │  Video/Frame │  (desde cámara o archivo)
  │  Input       │
  └──────┬───────┘
         │
         ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  1. PRE-PROCESAMIENTO                                       │
  │     - Leer frame (OpenCV)                                   │
  │     - Convertir BGR → RGB                                   │
  │     - Redimensionar a 224×224                               │
  │     - Normalizar con estadísticas ImageNet                  │
  │     - Convertir a tensor PyTorch                            │
  │     - Agregar dimensión de batch: (3, 224, 224) → (1, 3, 224, 224) │
  └─────────────────────────────────────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  2. CARGAR MODELO                                           │
  │     - Cargar checkpoint (.pth)                              │
  │     - Mover a dispositivo (CPU/CUDA/OAK-1)                  │
  │     - Poner en modo evaluación: model.eval()                │
  └─────────────────────────────────────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  3. INFERENCIA                                              │
  │     - Ejecutar forward pass                                 │
  │     - torch.no_grad() (no calcular gradientes)              │
  │     - Output: logits (1, 3)                                 │
  └─────────────────────────────────────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  4. POST-PROCESAMIENTO                                      │
  │     - Aplicar softmax para obtener probabilidades           │
  │     - argmax para obtener clase predicha                    │
  │     - Mapear índice a nombre de clase                       │
  │       0 → "Estado_0_Prefloracion"                           │
  │       1 → "Estado_1_Floracion_Intermedia"                   │
  │       2 → "Estado_2_Floracion_Maxima"                       │
  └─────────────────────────────────────────────────────────────┘
         │
         ▼
  ┌──────────────┐
  │  Predicción  │  (clase + confianza)
  │  + Confianza │
  └──────────────┘
```

### 8.2 Flujo Específico por Modelo

Aunque el flujo general es idéntico, los modelos difieren en lo que sucede dentro del bloque **"3. INFERENCIA"**:

#### EfficientNet-B0:
```
Input tensor (1, 3, 224, 224)
    ↓
[13 capas MBConv con depthwise separable convolutions]
    ↓
Global Average Pooling
    ↓
Dropout (p=0.3)
    ↓
Dense(1280 → 512) + ReLU + BatchNorm
    ↓
Dropout (p=0.15)
    ↓
Dense(512 → 3)
    ↓
Output logits (1, 3)
```

#### MobileNetV3-Small:
```
Input tensor (1, 3, 224, 224)
    ↓
[11 capas bottleneck con Hard-Swish]
    ↓
Global Average Pooling
    ↓
Dense(576 → 512) + Hard-Swish
    ↓
Dropout (p=0.2)
    ↓
Dense(512 → 3)
    ↓
Output logits (1, 3)
```

#### MobileNetV3-Large:
```
Input tensor (1, 3, 224, 224)
    ↓
[17 capas bottleneck con Hard-Swish]
    ↓
Global Average Pooling
    ↓
Dense(960 → 1280) + Hard-Swish
    ↓
Dropout (p=0.2)
    ↓
Dense(1280 → 3)
    ↓
Output logits (1, 3)
```

### 8.3 Ejemplo de Código de Inferencia

```python
import torch
from src.models.factory import create_model
from torchvision import transforms
import cv2
import numpy as np

# 1. Cargar modelo
model = create_model(
    model_name="mobilenet_v3_small",
    num_classes=3,
    pretrained=False,
    device="cpu"
)

# 2. Cargar pesos entrenados
checkpoint = torch.load("checkpoints/mobilenet_small/best_model.pth", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# 3. Definir transformaciones
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
])

# 4. Mapeo de clases
class_names = {
    0: "Estado_0_Prefloracion",
    1: "Estado_1_Floracion_Intermedia",
    2: "Estado_2_Floracion_Maxima"
}

# 5. Leer y procesar imagen
image = cv2.imread("frame.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_tensor = transform(image).unsqueeze(0)  # Agregar dimensión de batch

# 6. Inferencia
with torch.no_grad():
    outputs = model(image_tensor)
    probabilities = torch.softmax(outputs, dim=1)
    predicted_class = torch.argmax(outputs, dim=1).item()
    confidence = probabilities[0, predicted_class].item()

# 7. Resultado
print(f"Predicción: {class_names[predicted_class]}")
print(f"Confianza: {confidence:.2%}")
```

---

## 9. Recomendaciones para OAK-1

### 9.1 ¿Qué es OAK-1?

**OAK-1** es una cámara de profundidad con acelerador de IA integrado de Luxonis:

| Especificación | Valor |
|---------------|-------|
| **Procesador** | Intel Myriad X VPU |
| **Potencia IA** | ~1 TOPS (INT8) |
| **RAM** | 512 MB LPDDR4 |
| **Cámara** | 12 MP RGB |
| **Conexión** | USB 3.0 |
| **Precio** | ~$99 USD |

### 9.2 Restricciones de OAK-1

1. **Formato de modelo:** Requiere **OpenVINO** o **ONNX** (no PyTorch nativo)
2. **Cuantización:** Funciona mejor con **INT8** (no FP32)
3. **Operaciones soportadas:** No todas las operaciones de PyTorch están disponibles
4. **Memoria:** 512 MB compartido entre modelo + buffers de imagen

### 9.3 Recomendación Final

| Criterio | Modelo Recomendado | Razón |
|----------|-------------------|-------|
| **Máxima velocidad** | MobileNetV3-Small | 56M MACs = ~40-60 FPS en OAK-1 |
| **Mejor precisión** | EfficientNet-B0 | Más parámetros = mejor feature extraction |
| **Balance óptimo** | **MobileNetV3-Large** | 218M MACs = ~15-25 FPS, buena precisión |

### 9.4 Pasos para Deployment en OAK-1

```
┌─────────────────────────────────────────────────────────────┐
│  Pipeline de Deployment para OAK-1                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. ENTRENAR MODELO EN PYTORCH                              │
│     python src/training/train.py --model mobilenet_v3_small │
│                                                             │
│  2. EXPORTAR A ONNX                                         │
│     torch.onnx.export(model, dummy_input, "model.onnx")     │
│                                                             │
│  3. CUANTIZAR A INT8                                        │
│     Usar OpenVINO NNCF o Intel Post-Training Optimization   │
│                                                             │
│  4. CONVERTIR A BLOB (formato OAK)                          │
│     Usar DepthAI blobconverter                              │
│                                                             │
│  5. CARGAR EN OAK-1                                         │
│     Usar SDK de DepthAI (Python o C++)                      │
│                                                             │
│  6. TESTEAR EN DISPOSITIVO                                  │
│     Medir FPS real y latencia                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 10. Cómo Usar Este Sistema

### 10.1 Estructura del Proyecto

```
bonita-white-classifier/
├── src/
│   ├── models/
│   │   ├── factory.py                 ← Crear modelos por nombre
│   │   ├── efficientnet/
│   │   │   └── model.py               ← EfficientNet-B0
│   │   └── mobilenet/
│   │       └── model.py               ← MobileNetV3 (Small + Large)
│   ├── training/
│   │   ├── train.py                   ← Script de entrenamiento
│   │   └── evaluate.py                ← Script de evaluación
│   ├── data/
│   │   └── dataset.py                 ← Dataset y DataLoader
│   └── utils/
│       └── metrics.py                 ← Métricas y visualización
├── configs/
│   ├── efficientnet/
│   │   └── efficientnet_b0.yaml
│   └── mobilenet/
│       ├── mobilenet_v3_small.yaml
│       └── mobilenet_v3_large.yaml
├── scripts/
│   └── compare_models.py              ← Comparar los 3 modelos
├── checkpoints/                       ← Modelos entrenados
├── logs/                              ← Logs de entrenamiento
└── data/
    └── splits/                        ← Dataset (train/val/test)
```

### 10.2 Comandos de Entrenamiento

```bash
# Navegar al directorio del proyecto
cd bonita-white-classifier

# Entrenar MobileNetV3-Small
python src/training/train.py \
    --model mobilenet_v3_small \
    --config configs/mobilenet/mobilenet_v3_small.yaml

# Entrenar MobileNetV3-Large
python src/training/train.py \
    --model mobilenet_v3_large \
    --config configs/mobilenet/mobilenet_v3_large.yaml

# Entrenar EfficientNet-B0
python src/training/train.py \
    --model efficientnet_b0 \
    --config configs/efficientnet_b0.yaml

# Entrenar en modo debug (solo 5 batches)
python src/training/train.py \
    --model mobilenet_v3_small \
    --config configs/mobilenet/mobilenet_v3_small.yaml \
    --debug

# Comparar los 3 modelos automáticamente
python scripts/compare_models.py
```

### 10.3 Comandos de Evaluación

```bash
# Evaluar modelo en dataset de prueba
python src/training/evaluate.py \
    --checkpoint checkpoints/mobilenet_small/best_model.pth \
    --split test

# Evaluar en validación
python src/training/evaluate.py \
    --checkpoint checkpoints/mobilenet_small/best_model.pth \
    --split val
```

### 10.4 Argumentos Disponibles

#### train.py

| Argumento | Descripción | Default |
|-----------|-------------|---------|
| `--model` | Modelo a usar | `efficientnet_b0` |
| `--config` | Archivo de configuración | Automático según modelo |
| `--resume` | Checkpoint para continuar | `None` |
| `--debug` | Modo debug (5 batches) | `False` |
| `--seed` | Seed para reproducibilidad | `42` |

#### evaluate.py

| Argumento | Descripción | Default |
|-----------|-------------|---------|
| `--checkpoint` | Ruta al checkpoint | **Requerido** |
| `--split` | Split a evaluar | `test` |
| `--data-dir` | Directorio de datos | `data/splits` |
| `--output-dir` | Directorio de resultados | `results` |
| `--batch-size` | Batch size | `32` |
| `--device` | Dispositivo | `auto` |

---

## Apéndice A: Glosario de Términos

| Término | Definición |
|---------|------------|
| **MACs** | Multiply-Accumulate Operations. Medida de complejidad computacional. |
| **Transfer Learning** | Reutilizar modelo pre-entrenado en nueva tarea. |
| **Fine-tuning** | Ajustar pesos de modelo pre-entrenado con nueva data. |
| **Backbone** | Parte del modelo que extrae features (convoluciones). |
| **Classifier Head** | Capas finales que mapean features a clases. |
| **Batch Size** | Número de muestras procesadas antes de actualizar pesos. |
| **Learning Rate** | Magnitud de actualización de pesos en cada iteración. |
| **Dropout** | Técnica de regularización que "apaga" neuronas aleatoriamente. |
| **Early Stopping** | Detener entrenamiento si validación no mejora. |
| **Class Weights** | Pesos para compensar desbalance de clases. |
| **Softmax** | Función que convierte logits a probabilidades (suma 1). |
| **CrossEntropy** | Función de pérdida para clasificación. |
| **INT8/FP32** | Precision numérica: 8-bit integer vs 32-bit float. |

---

## Apéndice B: Referencias

1. **EfficientNet:** Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML.

2. **MobileNetV3:** Howard, A., et al. (2019). Searching for MobileNetV3. ICCV.

3. **OAK-1:** Luxonis. (2023). OAK-1 Documentation. https://docs.luxonis.com/

4. **PyTorch:** Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. NeurIPS.

---

## Apéndice C: Historial de Cambios

| Versión | Fecha | Cambios |
|---------|-------|---------|
| 1.0 | 2026-03-13 | Documentación inicial creada |

---

**Fin del Documento**

Para preguntas o actualizaciones, contactar al equipo de IA.