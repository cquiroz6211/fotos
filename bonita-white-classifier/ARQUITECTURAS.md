# Arquitecturas de los Modelos Propuestos

## ¿Por qué estas dos arquitecturas?

El sistema debe correr en una cámara OAK-1 con hardware muy limitado (~1 TOPS, 512 MB RAM). Eso descarta redes grandes. **EfficientNet-B0** y **MobileNetV3-Small** son dos de las arquitecturas más eficientes disponibles para clasificación de imágenes, y ambas fueron diseñadas explícitamente para dispositivos con recursos restringidos.

La pregunta que guía la comparación es: ¿cuánta precisión ganamos con EfficientNet-B0 respecto a MobileNetV3-Small, y a qué costo computacional?

---

## 1. EfficientNet-B0

### ¿Qué idea tiene detrás?

La mayoría de las redes se hacen más precisas haciéndolas más profundas (más capas), más anchas (más filtros) o aumentando la resolución de entrada. EfficientNet propone hacer las tres cosas **a la vez y en proporción**, usando un único coeficiente de escala $\phi$:

$$d = \alpha^\phi \quad w = \beta^\phi \quad r = \gamma^\phi$$

Para EfficientNet-B0, $\phi = 1$ con $\alpha=1.2$, $\beta=1.1$, $\gamma=1.15$. Es el modelo base de la familia — el más pequeño — pero ya incorpora el principio de escala compuesta.

### Bloque fundamental: MBConv

Cada etapa de la red repite un bloque llamado **MBConv** (Mobile Inverted Bottleneck). La idea es procesar la imagen en un espacio expandido (más canales), aplicar la convolución ahí, y volver a comprimir. Esto es más eficiente que una convolución estándar.

```mermaid
flowchart TD
    A([Input]) --> B["Expansion Conv 1×1\nBN + Swish\nExpande los canales"]
    B --> C["Depthwise Conv k×k\nBN + Swish\nUna convolución por canal"]
    C --> D["SE Block\nAtención por canal\nGlobalAvgPool → FC → Sigmoid → Scale"]
    D --> E["Projection Conv 1×1\nBN\nComprime los canales de vuelta"]
    E --> F{¿stride=1 y\nmismos canales?}
    F -- Sí --> G(["Suma con Input\nSkip connection"])
    F -- No --> H([Output])
```

El **bloque SE** (Squeeze-Excitation) es lo que diferencia a EfficientNet de redes más simples: aprende a darle más importancia a ciertos canales (características) y menos a otros, funcionando como un mecanismo de atención sobre los filtros.

La activación **Swish** también es distintiva:

$$\text{Swish}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

Es suave y no tiene un límite superior, lo que ayuda al flujo de gradientes durante el entrenamiento.

### Arquitectura completa adaptada

```mermaid
flowchart TD
    A(["Imagen de entrada\n224 × 224 × 3"]) --> B["Conv 3×3, stride=2\nBN + Swish\n→ 112×112"]
    B --> C["MBConv1 ×1\n→ 112×112"]
    C --> D["MBConv6 ×2\n→ 56×56"]
    D --> E["MBConv6 ×2\n→ 28×28"]
    E --> F["MBConv6 ×3\n→ 14×14"]
    F --> G["MBConv6 ×3\n→ 14×14"]
    G --> H["MBConv6 ×4\n→ 7×7"]
    H --> I["MBConv6 ×1\n→ 7×7"]
    I --> J["Conv 1×1, BN + Swish\n→ 1280 canales"]
    J --> K["Global Average Pooling\n→ vector 1280-dim"]
    K --> L["Classifier Head\n(reemplazado)"]
    L --> M(["Salida: 3 clases"])
```

### Classifier head

El head original de EfficientNet fue reemplazado por uno más robusto para nuestro dominio:

```mermaid
flowchart TD
    A(["Features del backbone\n1280-dim"]) --> B["Dropout p=0.30"]
    B --> C["Linear 1280 → 512"]
    C --> D["ReLU"]
    D --> E["BatchNorm1d 512"]
    E --> F["Dropout p=0.15"]
    F --> G["Linear 512 → 3"]
    G --> H(["Logits\nPrefloración / Intermedia / Máxima"])
```

El dropout en dos etapas (0.30 → 0.15) aplica más regularización en la transición desde el backbone y menos antes de la clasificación final. BatchNorm1d estabiliza la distribución de activaciones en las 512 dimensiones intermedias.

### Especificaciones

| Parámetro | Valor |
|-----------|-------|
| Parámetros totales | 5.3 M |
| MACs por inferencia | 390 M |
| Tamaño (FP32) | 56.5 MB |
| Resolución de entrada | 224 × 224 |
| Activación | Swish |
| Dropout | 0.30 / 0.15 (dos etapas) |

---

## 2. MobileNetV3-Small

### ¿Qué idea tiene detrás?

MobileNetV3-Small fue diseñado con un objetivo muy concreto: **mínima latencia en hardware móvil**. Para lograrlo combina dos estrategias:

1. **Búsqueda de arquitectura automática (NAS)**: un algoritmo busca la combinación óptima de capas para maximizar exactitud por unidad de latencia.
2. **Ajuste manual** de las capas de entrada y salida, que NAS tiende a hacer ineficientes.

El resultado es una red que usa 7 veces menos operaciones que EfficientNet-B0, manteniendo una precisión comparable o superior en tareas de dominio específico.

### Bloque fundamental: Bottleneck con SE selectivo

Comparte la estructura MBConv de EfficientNet, pero el módulo SE solo está presente en algunos bloques (no en todos), reduciendo el costo computacional. Además, reemplaza Swish por **HardSwish**:

```mermaid
flowchart TD
    A([Input]) --> B["Expansion Conv 1×1\nBN + HardSwish / ReLU"]
    B --> C["Depthwise Conv k×k\nBN + HardSwish / ReLU"]
    C --> D{¿Bloque con SE?}
    D -- Sí --> E["SE Block\nAvgPool → FC → ReLU → FC → HardSigmoid → Scale"]
    D -- No --> F["Projection Conv 1×1\nBN"]
    E --> F
    F --> G{¿stride=1?}
    G -- Sí --> H(["Suma con Input\nSkip connection"])
    G -- No --> I([Output])
```

**HardSwish** es una aproximación lineal por partes de Swish que evita calcular exponenciales:

$$\text{HardSwish}(x) = x \cdot \frac{\text{ReLU6}(x + 3)}{6}$$

En hardware sin soporte nativo para operaciones exponenciales (como el Myriad X de OAK-1), esto se traduce en una aceleración real de inferencia.

### Arquitectura completa adaptada

```mermaid
flowchart TD
    A(["Imagen de entrada\n224 × 224 × 3"]) --> B["Conv 3×3, stride=2\nBN + HardSwish\n→ 112×112"]
    B --> C["Bottleneck ×1\n(sin SE)\n→ 56×56"]
    C --> D["SE-Bottleneck ×1\n→ 28×28"]
    D --> E["Bottleneck ×2\n→ 14×14"]
    E --> F["SE-Bottleneck ×2\n→ 14×14"]
    F --> G["SE-Bottleneck ×1\n→ 14×14"]
    G --> H["Bottleneck ×2\n→ 7×7"]
    H --> I["Conv 1×1, BN + HardSwish\n→ 576 canales"]
    I --> J["Global Average Pooling\n→ vector 576-dim"]
    J --> K["Classifier Head\n(reemplazado)"]
    K --> L(["Salida: 3 clases"])
```

La red termina con solo **576 dimensiones** antes del head (vs 1280 de EfficientNet). Esto es posible porque cada bloque fue optimizado para retener solo la información más relevante.

### Classifier head

```mermaid
flowchart TD
    A(["Features del backbone\n576-dim"]) --> B["Linear 576 → 512"]
    B --> C["HardSwish"]
    C --> D["Dropout p=0.20"]
    D --> E["Linear 512 → 3"]
    E --> F(["Logits\nPrefloración / Intermedia / Máxima"])
```

El head usa HardSwish en lugar de ReLU para mantener consistencia con la activación del backbone, evitando discontinuidades en la distribución de gradientes durante el fine-tuning. Es más simple que el head de EfficientNet — no necesita BatchNorm intermedio porque la red base ya es más estable.

### Especificaciones

| Parámetro | Valor |
|-----------|-------|
| Parámetros totales | 2.5 M |
| MACs por inferencia | 56 M |
| Tamaño (FP32) | 14.9 MB |
| Resolución de entrada | 224 × 224 |
| Activación | HardSwish |
| Dropout | 0.20 (una etapa) |

---

## 3. Comparación directa

```mermaid
flowchart LR
    subgraph EfficientNet-B0
        direction TB
        E1["Entrada 224×224"] --> E2["8 etapas MBConv\n+ SE en todas"]
        E2 --> E3["1280-dim features"]
        E3 --> E4["Head: Dropout→512→BN→Dropout→3"]
        E4 --> E5["5.3M params\n390M MACs\n56.5MB"]
    end

    subgraph MobileNetV3-Small
        direction TB
        M1["Entrada 224×224"] --> M2["7 bloques Bottleneck\n+ SE selectivo"]
        M2 --> M3["576-dim features"]
        M3 --> M4["Head: 512→HardSwish→Dropout→3"]
        M4 --> M5["2.5M params\n56M MACs\n14.9MB"]
    end
```

| Criterio | EfficientNet-B0 | MobileNetV3-Small |
|----------|:--------------:|:-----------------:|
| Parámetros | 5.3 M | **2.5 M** |
| MACs | 390 M | **56 M** |
| Tamaño modelo | 56.5 MB | **14.9 MB** |
| Activación backbone | Swish | HardSwish |
| SE en todos los bloques | Sí | Solo en algunos |
| Latencia en CPU | 42.78 ms | **7.57 ms** |
| Accuracy (test) | 90.52% | **91.38%** |
| F1-Score Macro | 0.881 | **0.902** |

MobileNetV3-Small requiere **7 veces menos operaciones** y es **5.7 veces más rápido**, y aun así obtiene mejor F1-Score. Esto se explica porque con un dataset pequeño (761 imágenes), un modelo más compacto generaliza mejor: tiene menos capacidad para memorizar el conjunto de entrenamiento.

---

## 4. Transfer Learning: punto de partida compartido

Ambas arquitecturas se inicializan con **pesos preentrenados en ImageNet-1K** (1.28 millones de imágenes, 1000 clases). Esto significa que los filtros del backbone ya saben detectar bordes, texturas, formas y patrones generales antes de ver una sola imagen de *Callistephus chinensis*.

```mermaid
flowchart LR
    A["ImageNet\n1.28M imágenes\n1000 clases"] -->|Preentrenamiento| B["Backbone\nfiltros generales"]
    B -->|Reemplazar head| C["Head nuevo\n3 clases fenológicas"]
    C -->|Fine-tuning completo\n761 imágenes| D["Modelo final\nadaptado al dominio"]
```

Se aplica **fine-tuning completo** (sin congelar capas): todos los pesos se actualizan durante el entrenamiento. Esto es apropiado cuando el dataset es pequeño y las clases son muy específicas de dominio — las texturas florales de *Callistephus chinensis* no tienen correspondencia directa con categorías ImageNet.
