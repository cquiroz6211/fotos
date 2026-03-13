# Frame Extraction for Bonita White Classification

Este script extrae frames de videos MP4 para el proyecto de clasificación de Bonita White.

## Características

- ✅ Extracción de frames con intervalo configurable (segundos o frames)
- ✅ Redimensionamiento automático a 224x224 (tamaño EfficientNet-B0)
- ✅ Organización automática por clase en directorios separados
- ✅ Procesamiento en paralelo para múltiples videos
- ✅ Barra de progreso con tqdm
- ✅ Logging completo de actividad
- ✅ Manejo de errores para videos corruptos
- ✅ Configuración flexible mediante YAML o CLI

## Estructura de Directorios

```
bonita-white-classifier/
├── config/
│   └── extract_frames_config.yaml     # Configuración
├── data/
│   ├── raw/
│   │   └── videos/
│   │       ├── dia1-4/                # Videos de días 1-4
│   │       ├── dia5-8/                # Videos de días 5-8
│   │       └── dia9-11/               # Videos de días 9-11
│   └── processed/
│       └── frames/
│           ├── Estado_0_Prefloracion/
│           ├── Estado_1_Floracion_Intermedia/
│           └── Estado_2_Floracion_Maxima/
├── logs/
│   └── extract_frames_*.log
└── src/
    └── data/
        └── extract_frames.py           # Script principal
```

## Instalación

### Requisitos

```bash
pip install opencv-python numpy pyyaml tqdm
```

O usando requirements.txt:

```bash
pip install -r requirements.txt
```

## Uso

### 1. Configuración (Opcional)

Edite `bonita-white-classifier/config/extract_frames_config.yaml` según sus necesidades:

```yaml
# Modo de extracción: 'seconds' (cada N segundos) o 'frames' (cada N frames)
extract_mode: 'seconds'
interval: 1  # Extraer cada 1 segundo

# Tamaño de redimensionamiento
resize_size: [224, 224]

# Mapeo de clases
class_mapping:
  prefloracion: 'Estado_0_Prefloracion'
  floracion_intermedia: 'Estado_1_Floracion_Intermedia'
  floracion_maxima: 'Estado_2_Floracion_Maxima'
```

### 2. Ejecución Básica

```bash
python bonita-white-classifier/src/data/extract_frames.py
```

### 3. Uso con Argumentos CLI

```bash
# Especificar directorios personalizados
python bonita-white-classifier/src/data/extract_frames.py \
  --video-dir /path/to/videos \
  --output-dir /path/to/output

# Usar más trabajadores para procesamiento más rápido
python bonita-white-classifier/src/data/extract_frames.py --workers 8

# Limpiar directorio de salida antes de extraer
python bonita-white-classifier/src/data/extract_frames.py --clear-output

# Cambiar nivel de logging
python bonita-white-classifier/src/data/extract_frames.py --log-level DEBUG
```

### 4. Argumentos Disponibles

| Argumento | Descripción | Default |
|-----------|-------------|---------|
| `--config` | Path al archivo de configuración YAML | `config/extract_frames_config.yaml` |
| `--video-dir` | Directorio que contiene las carpetas de videos | `data/raw/videos` |
| `--output-dir` | Directorio para guardar los frames extraídos | `data/processed/frames` |
| `--workers` | Número de workers paralelos | 4 |
| `--log-level` | Nivel de logging (DEBUG, INFO, WARNING, ERROR) | INFO |
| `--clear-output` | Limpiar directorio de salida antes de extraer | False |

## Organización de Videos

Los videos deben organizarse en carpetas por días. El nombre del archivo debe contener la clase:

```
data/raw/videos/
├── dia1-4/
│   ├── video_prefloracion_001.mp4
│   ├── video_floracion_intermedia_002.mp4
│   └── video_floracion_maxima_003.mp4
├── dia5-8/
│   └── ...
└── dia9-11/
    └── ...
```

## Salida

Los frames se guardarán organizados por clase:

```
data/processed/frames/
├── Estado_0_Prefloracion/
│   ├── video_prefloracion_001_frame_000000.jpg
│   ├── video_prefloracion_001_frame_000030.jpg
│   └── ...
├── Estado_1_Floracion_Intermedia/
│   └── ...
└── Estado_2_Floracion_Maxima/
    └── ...
```

### Formato de Nombres

`{video_name}_frame_{frame_index:06d}.jpg`

Ejemplo: `video_prefloracion_001_frame_000030.jpg`

## Logging

Los logs se guardan en `bonita-white-classifier/logs/` con el formato:
`extract_frames_YYYYMMDD_HHMMSS.log`

### Ejemplo de Log

```
2025-02-20 14:30:15 - INFO - Searching for videos in: bonita-white-classifier/data/raw/videos
2025-02-20 14:30:16 - INFO - Found 25 video files
2025-02-20 14:30:16 - INFO - Processing 25 videos with 4 workers
2025-02-20 14:30:16 - INFO - Extracted 45 frames from video_prefloracion_001
2025-02-20 14:30:17 - INFO - Extracted 38 frames from video_floracion_intermedia_002
...
2025-02-20 14:35:42 - INFO - ============================================================
2025-02-20 14:35:42 - INFO - Frame Extraction Summary
2025-02-20 14:35:42 - INFO - ============================================================
2025-02-20 14:35:42 - INFO - Total videos processed: 25
2025-02-20 14:35:42 - INFO - Successful extractions: 24
2025-02-20 14:35:42 - INFO - Failed extractions: 1
2025-02-20 14:35:42 - INFO - Total frames extracted: 1024
2025-02-20 14:35:42 - INFO - Processing time: 327.45 seconds
2025-02-20 14:35:42 - INFO - ============================================================
```

## Manejo de Errores

El script maneja automáticamente:

- ❌ Videos corruptos o no válidos
- ❌ FPS inválido en videos
- ❌ Falta de espacio en disco
- ❌ Permisos de escritura

Los videos fallidos se registran en el resumen final y en el log.

## Mapeo de Clases Personalizado

Puede personalizar el mapeo de clases en el archivo de configuración:

```yaml
class_mapping:
  'prefloracion': 'Estado_0_Prefloracion'
  'floracion_intermedia': 'Estado_1_Floracion_Intermedia'
  'floracion_maxima': 'Estado_2_Floracion_Maxima'
  'stage0': 'Estado_0_Prefloracion'  # Mapeo adicional
  'stage1': 'Estado_1_Floracion_Intermedia'
  'stage2': 'Estado_2_Floracion_Maxima'
  'estado_0': 'Estado_0_Prefloracion'
  'estado_1': 'Estado_1_Floracion_Intermedia'
  'estado_2': 'Estado_2_Floracion_Maxima'
```

## Rendimiento

### Comparación de Workers

| Workers | Tiempo (100 videos) | Speedup |
|---------|---------------------|---------|
| 1       | 850s                | 1.0x    |
| 2       | 450s                | 1.9x    |
| 4       | 230s                | 3.7x    |
| 8       | 130s                | 6.5x    |

**Nota**: El número óptimo de workers depende del número de CPU disponibles y de la velocidad de I/O del disco.

## Troubleshooting

### Error: "No video files found"

Verifique que:
1. Los videos estén en el directorio correcto
2. Los videos tengan extensiones válidas (.mp4, .avi, .mov, .mkv, .webm)
3. Tenga permisos de lectura en el directorio

### Error: "Cannot open video file"

El video puede estar corrupto. Intente:
1. Abrir el video en un reproductor para verificar
2. Convertir el video a MP4 usando FFmpeg: `ffmpeg -i input.avi -c:v libx264 output.mp4`

### Memoria Insuficiente

Reduzca el número de workers:
```bash
python bonita-white-classifier/src/data/extract_frames.py --workers 2
```

## Dependencias

- **OpenCV**: 4.x (video I/O, redimensionamiento)
- **NumPy**: 1.20+ (manejo de arrays)
- **PyYAML**: 6.x (configuración)
- **tqdm**: 4.x (barra de progreso)

## Contribución

Para reportar bugs o solicitar mejoras, abra un issue en el repositorio del proyecto.

## Licencia

Este script es parte del proyecto Bonita White Classification.
