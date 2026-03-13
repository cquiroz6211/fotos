# Bonita White Classifier - Comandos Útiles

Este documento contiene los comandos principales disponibles para ejecutar y visualizar las diferentes interfaces y scripts del proyecto.

## Requisitos Previos

Antes de ejecutar cualquier comando web de Streamlit, es importante asegurarse de tener instaladas las dependencias del proyecto. Para ello, ejecuta desde tu terminal:

```bash
pip install -r requirements.txt
```
*(Es recomendable usar un entorno virtual como `venv` o `conda` antes de instalar las dependencias).*

---

##  Aplicaciones y Vistas en Streamlit

El proyecto cuenta con varias interfaces construidas con **Streamlit**. Para ejecutarlas, debes abrir una terminal, navegar hasta la carpeta raíz del proyecto (`c:\Users\Administrador\Documents\fotos\bonita-white-classifier`) y ejecutar alguno de los siguientes comandos:

### 1. Ejecutar el Analyzer
Si quieres ver la vista principal del analizador de video:
```bash
streamlit run video_analyzer.py
```

### 2. Ejecutar Video Analytics
Si necesitas acceder a la herramienta de análisis estadístico/reportes del video:
```bash
streamlit run video_analytics.py
```

### 3. Ejecutar el Demo general
Para lanzar el demo completo implementado para demostraciones:
```bash
streamlit run demo.py
```

Una vez que corras alguno de estos comandos en terminal:
- Aparecerá un servidor local (normalmente `http://localhost:8501`).
- Tu navegador por defecto se abrirá automáticamente a la vista de la aplicación elegida.

---

## 🛠️ Otros Scripts de Utilidad

Si necesitas correr los scripts adicionales definidos en el directorio raíz (por ejemplo, para extraer frames o validaciones), usa Python normalmente:

### Extraer Frames y Preparación
```bash
python setup_extract_frames.py
```

### Pruebas de Extracción de frames
```bash
python test_extract_frames.py
```
