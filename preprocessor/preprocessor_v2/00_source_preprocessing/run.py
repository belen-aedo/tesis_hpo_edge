"""
Registro y punto de entrada del módulo de preprocesamiento de fuentes.

Uso desde la raíz del proyecto:
    python preprocessor_v2/00_source_preprocessing/run.py

O importando directamente:
    from preprocessor_v2.source_preprocessing import run_all
"""

from __future__ import annotations

import sys
from pathlib import Path

# Asegurar que config.py sea importable desde cualquier CWD
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_config

from bot_iot_full import BotIoTFullPreprocessor
from ciciot2023 import CICIoTDataset2023Preprocessor
from ds2os import DS2OSPreprocessor
from iot23 import IOT23Preprocessor
from iotid20 import IoTID20Preprocessor
from n_baiot import NBaIoTPreprocessor
from ton_iot import TonIoTPreprocessor

# Registro: clave config.json → clase preprocesadora
PREPROCESSOR_REGISTRY: dict[str, type] = {
    "N-baiot":            NBaIoTPreprocessor,
    "IoTID20":            IoTID20Preprocessor,
    "Bot-IoT-full":       BotIoTFullPreprocessor,
    "DS2OS":              DS2OSPreprocessor,
    "Ton-IoT":            TonIoTPreprocessor,
    # Misma fuente con opciones de particion y codificacion mejoradas (config.json).
    "Ton-IoT-v2":         TonIoTPreprocessor,
    # IOT23 tiene tres variantes del mismo proceso de lectura de archivos .labeled;
    # la fracción de muestreo se controla en config.json via 'sample_fraction'.
    "IOT23":              IOT23Preprocessor,
    "IOT23-10percent":    IOT23Preprocessor,
    "IOT23-1percent":     IOT23Preprocessor,
    "CICIoTDataset2023":  CICIoTDataset2023Preprocessor,
}


def run_all(datasets_to_run: list[str] | None = None) -> None:
    """
    Ejecuta los preprocesadores de fuente para los datasets indicados.
    Si datasets_to_run es None, usa la lista de config.yaml (datasets_to_process).
    """
    conf = load_config()
    if not conf:
        return

    targets = datasets_to_run or conf["DATASETS_TO_PROCESS"]

    for name in targets:
        if name not in PREPROCESSOR_REGISTRY:
            print(f"⚠️  No existe preprocesador de fuente para '{name}'. Omitiendo.")
            continue
        if name not in conf["DATASETS"]:
            print(f"⚠️  Dataset '{name}' no encontrado en config.json. Omitiendo.")
            continue

        preprocessor_cls = PREPROCESSOR_REGISTRY[name]
        preprocessor = preprocessor_cls(dataset_name=name, config=conf)
        preprocessor.run()


if __name__ == "__main__":
    # Permite ejecutar un subset pasando nombres como argumentos:
    #   python run.py DS2OS N-baiot
    datasets = sys.argv[1:] if len(sys.argv) > 1 else None
    run_all(datasets)
