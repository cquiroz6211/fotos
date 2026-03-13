# Resultados de Comparacion de Modelos

**Fecha**: 13 de Marzo, 2026
**Proyecto**: Clasificador de Estados Fenologicos - Bonita White
**Objetivo**: Seleccionar el mejor modelo para deployment en OAK-1

---

## Resumen Ejecutivo

**Ganador: MobileNetV3-Small**

- Mejor accuracy (91.38% vs 90.52%)
- 5.7x mas rapido (7.57ms vs 42.78ms)
- 3.8x mas pequeno (14.9MB vs 56.5MB)

---

## Metricas Globales

| Metrica | EfficientNet-B0 | MobileNetV3-Small | Diferencia |
|---------|-----------------|-------------------|------------|
| Accuracy | 90.52% | **91.38%** | +0.86% |
| F1-Score (Macro) | 0.8808 | **0.9020** | +0.0212 |
| Precision (Macro) | **0.9267** | 0.9145 | -0.0122 |
| Recall (Macro) | 1.0000 | 1.0000 | 0 |
| Tiempo inferencia (ms) | 42.78 | **7.57** | -35.21ms |

---

## Explicacion de Metricas

### Accuracy (Precisión General)
- **Qué mide**: Porcentaje de predicciones correctas sobre el total.
- **Ejemplo**: 91.38% significa que acierta 91 de cada 100 flores.

### F1-Score (Macro)
- **Qué mide**: Balance entre precision y recall. Promedio de las 3 clases.
- **Rango**: 0 a 1 (1 = perfecto).
- **Por qué importa**: Un modelo equilibrado tiene F1 alto.

### Precision (Macro)
- **Qué mide**: De todas las veces que el modelo predijo una clase, cuántas estaba correcto.
- **Ejemplo**: Precision 0.91 = cuando dice "Estado 2", acierta el 91% de las veces.

### Recall (Macro)
- **Qué mide**: De todas las flores reales de una clase, cuántas detectó el modelo.
- **Ejemplo**: Recall 1.0 = detectó TODAS las flores de esa clase.

### Tiempo de Inferencia
- **Qué mide**: Milisegundos para procesar UNA imagen.
- **Importancia**: Menos tiempo = más FPS (fotogramas por segundo).

---

## Metricas por Clase

### Estado 0: Prefloracion

| Metrica | EfficientNet-B0 | MobileNetV3-Small |
|---------|-----------------|-------------------|
| Precision | 1.0000 | 1.0000 |
| Recall | 1.0000 | 1.0000 |
| F1-Score | 1.0000 | 1.0000 |

**Interpretacion**: Ambos modelos detectan perfectamente las flores en estado prefloracion.

---

### Estado 1: Floracion Intermedia

| Metrica | EfficientNet-B0 | MobileNetV3-Small |
|---------|-----------------|-------------------|
| Precision | 0.7800 | **1.0000** |
| Recall | **1.0000** | 0.7436 |
| F1-Score | 0.8764 | **0.8529** |

**Interpretacion**:
- EfficientNet detecta TODAS las intermedias pero tiene falsos positivos.
- MobileNetV3 tiene menos falsos positivos pero pierde algunas intermedias.

---

### Estado 2: Floracion Maxima

| Metrica | EfficientNet-B0 | MobileNetV3-Small |
|---------|-----------------|-------------------|
| Precision | **1.0000** | 0.7436 |
| Recall | 0.6207 | **1.0000** |
| F1-Score | 0.7660 | **0.8529** |

**Interpretacion**:
- EfficientNet es preciso pero pierde el 38% de las flores maximas.
- MobileNetV3 detecta TODAS las maximas pero tiene algunos falsos positivos.

---

## Matrices de Confusion

### EfficientNet-B0

```
              Pred_0  Pred_1  Pred_2
Real_0          48       0       0
Real_1           0      39       0
Real_2           0      11      18
```

**Problema**: Confunde Estado_2 con Estado_1 (11 errores)

---

### MobileNetV3-Small

```
              Pred_0  Pred_1  Pred_2
Real_0          48       0       0
Real_1           0      29      10
Real_2           0       0      29
```

**Problema**: Confunde Estado_1 con Estado_2 (10 errores)

---

## Comparacion para OAK-1

| Criterio | EfficientNet-B0 | MobileNetV3-Small | Ganador |
|----------|-----------------|-------------------|---------|
| Accuracy | 90.52% | **91.38%** | MobileNetV3 |
| Velocidad | 42.78ms (23 FPS) | **7.57ms (132 FPS)** | MobileNetV3 |
| Tamano | 56.5 MB | **14.9 MB** | MobileNetV3 |
| MACs | ~390M | **~60M** | MobileNetV3 |
| Complejidad | Alta | **Baja** | MobileNetV3 |

---

## Recomendacion Final

**Modelo seleccionado: MobileNetV3-Small**

### Razones:

1. **Mejor accuracy** (91.38% vs 90.52%)
2. **5.7x mas rapido** - critico para video en tiempo real
3. **3.8x mas pequeno** - cabe facilmente en OAK-1
4. **Menor consumo de energia** - importante para deployment en campo

### Proximos pasos:

1. Exportar a ONNX
2. Cuantizar a INT8
3. Crear pipeline DepthAI para OAK-1
4. Probar en hardware real

---

## Archivos Relevantes

| Archivo | Descripcion |
|---------|-------------|
| `checkpoints/mobilenet_small/best_model.pth` | Modelo entrenado |
| `checkpoints/best_model.pth` | EfficientNet-B0 entrenado |
| `scripts/evaluate_comparison.py` | Script de evaluacion |
| `configs/mobilenet/mobilenet_v3_small.yaml` | Config de entrenamiento |

---

*Generado automaticamente por el sistema de comparacion de modelos.*