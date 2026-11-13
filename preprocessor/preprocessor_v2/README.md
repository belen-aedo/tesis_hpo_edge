# preprocessor_v2 — Pipeline de Preprocesamiento de ML Robusto

Esta carpeta contiene la versión 2.0 del pipeline de preprocesamiento, rediseñada para ser modular, eficiente en memoria y, sobre todo, para garantizar la **ausencia total de Data Leakage (fuga de datos)** entre los conjuntos de entrenamiento y prueba.

## 🚀 Características Principales

- **No Data Leakage**: El escalado e imputación se ajustan (`fit`) exclusivamente con datos de entrenamiento.
- **Procesamiento Out-of-Core**: Capacidad para manejar datasets de más de 20GB (como IOT23) mediante procesamiento por trozos (`chunksize`).
- **Muestre Estratificado**: Generación de subconjuntos del 10% preservando la distribución de clases original (`category + sub_category`).
- **Selección de Características Multivariante**: Incluye VIF, Chi2, Mutual Info, Importancia LGBM y L1-Lasso.
- **Benchmark Automatizado**: Evaluación rápida con Decision Trees, Random Forest y Regresión Logística.
- **Monitoreo en Tiempo Real**: Panel de control interactivo para seguir la ejecución.

## 🛠 Estructura del Pipeline

La ejecución es secuencial y cada paso guarda sus resultados en formato **Apache Parquet**:

1.  `01_train_test_split.py`: Divide el source CSV en `train_raw` y `test_raw` (80/20).
2.  `02_preprocessing.py`: Aplica limpieza, imputación y escalado (MinMax). Las estadísticas se aprenden solo del Train.
3.  `03_stratified_subset.py`: Crea muestras reducidas (10%) para experimentación rápida.
4.  `04_feature_selection.py`: Selecciona las mejores características basándose en la muestra estratificada para mayor velocidad.
5.  `05_ml_benchmark.py`: Entrena modelos ligeros y guarda resultados en `RESULTS/preprocessor_v2_benchmark/`.

## 💻 Uso

### Ejecutar todo el pipeline
Para todos los datasets configurados (en segundo plano):
```bash
nohup bash preprocessor_v2/run_pipeline.sh > pipeline.log 2>&1 &
```

### Ejecutar datasets específicos
```bash
bash preprocessor_v2/run_pipeline.sh IoTID20 N-baiot
```

### Monitoreo en Tiempo Real (Modo Watch)
```bash
python3 preprocessor_v2/monitor.py --watch 5
```

## ⚙️ Configuración

- `config.yaml`: Define el entorno de ejecución (`high_performance`), rutas de `ROOT` y `RESULTS`, y el `chunksize`.
- `config.json`: Metadatos de cada dataset (nombre de etiquetas, columnas numéricas, rutas de origen).

## 🧪 Validación

Se incluye un script de validación de integridad (`validate_outputs.py`) que verifica:
- Inexistencia de filas duplicadas entre train y test.
- Consistencia en el número de características.
- Preservación de la distribución de clases.

Para correr los tests unitarios:
```bash
python3 -m pytest preprocessor_v2/tests/test_pipeline.py
```
