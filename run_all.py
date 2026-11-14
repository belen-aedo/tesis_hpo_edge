"""
run_all.py - ejecuta las etapas del proyecto en orden (Windows, Linux o Raspberry).

Etapas (en este orden):
    preprocesar   pasos 00-02 del preprocesador (CSV -> Parquet)             [PC]
    splits        1_prepare_splits.py : bloques train / val / test                [PC]
    diagnostico   2_diagnose.py       : ¿sirven las metricas? (opcional)          [PC]
    optimizar     3_optimize.py --all : Optuna + oraculo + linea base             [PC]
    borde         4_edge_benchmark.py : latencia / CPU / RAM                      [RASPBERRY]
    evaluar       5_evaluate.py       : evaluacion final sobre test_final         [PC]
    reporte       6_report.py         : tablas LaTeX                              [PC]

Por defecto corre:  preprocesar, splits, optimizar, evaluar, reporte
('borde' se corre en la Raspberry; 'diagnostico' es opcional).

EJEMPLOS
    python run_all.py --rapido                       # prueba corta de todo
    python run_all.py --etapas splits,optimizar
    python run_all.py --etapas borde                 # en la Raspberry
    python run_all.py --etapas evaluar,reporte       # al volver a la PC
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Carpetas del proyecto y lista de etapas
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
PRE_V2 = ROOT / "preprocessor" / "preprocessor_v2"
CSV_REL = "Datasets/Sources/TON-IoT/Train_Test_datasets/Train_Test_Network_dataset"
DEFAULT = ["preprocesar", "splits", "optimizar", "evaluar", "reporte"]
ALL = ["preprocesar", "splits", "diagnostico", "optimizar", "borde", "evaluar", "reporte"]


# Ejecuta un comando, muestra cuanto demoro y detiene todo si falla
def run(cmd: list[str], cwd: Path, env_extra: dict | None = None) -> None:
    env = dict(os.environ, **(env_extra or {}))
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    t0 = time.time()
    r = subprocess.run([str(c) for c in cmd], cwd=str(cwd), env=env)
    if r.returncode != 0:
        sys.exit(f"\nFALLO (codigo {r.returncode}) en: {' '.join(str(c) for c in cmd)}")
    print(f"   ok en {time.time() - t0:.1f}s", flush=True)


# Corre los pasos del preprocesador
def etapa_preprocesar(cfg: dict) -> None:
    if not PRE_V2.exists():
        sys.exit("Falta el preprocesador en preprocessor/ (vea LEEME en esa carpeta).")
    run([sys.executable, ROOT / "scripts" / "0_configurar_preprocessor.py"], ROOT)
    # Protege el CSV original: el paso 00 lee 'Train_Test_Labels.csv' si existe (si no, lee su
    # propio archivo de salida) y escribe 'preprocessed.csv' (la ruta "path" de config.json),
    # agregandole la columna `category`. Con la copia, nunca lee y escribe el mismo archivo.
    folder = ROOT / CSV_REL
    net, lab = folder / "Train_Test_Network.csv", folder / "Train_Test_Labels.csv"
    if not net.exists() and not lab.exists():
        sys.exit(f"Falta el dataset: copie Train_Test_Network.csv en {folder}")
    if not lab.exists():
        print(f"copia de respaldo del CSV original -> {lab.name}")
        shutil.copy2(net, lab)
    # PYTHONUTF8: el codigo del preprocesador imprime emojis y lee config.yaml con open() sin
    # codificacion; en Windows (cp1252) eso falla si la salida se redirige o hay tildes.
    env = {"KMP_DUPLICATE_LIB_OK": "True", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    # Cada paso del preprocessor se ejecuta para el dataset de la configuracion
    # (Ton-IoT = version 1; Ton-IoT-v2 = deduplicada, estratificada, log y one-hot completo)
    ds = cfg["dataset"]["name"]
    pasos = ["00_source_preprocessing/run.py", "01_train_test_split.py", "02_preprocessing.py"]
    # El paso 03 (muestra del 10 %) solo hace falta con input.variant: stratified
    if cfg["input"]["variant"] == "stratified":
        pasos.append("03_stratified_subset.py")
    # El paso 04 solo hace falta si se usan las columnas elegidas por el paso 04
    if cfg["features"]["mode"] == "selector":
        pasos.append("04_feature_selection.py")
    for p in pasos:
        run([sys.executable, p, ds], PRE_V2, env)


# Lee las opciones y corre las etapas pedidas, en orden
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--etapas", default=",".join(DEFAULT),
                    help=f"lista separada por comas de: {', '.join(ALL)}")
    ap.add_argument("--config", default=None, help="yaml que se mezcla sobre config/experiment.yaml")
    ap.add_argument("--rapido", action="store_true", help="optimize con 25 trials y 1 semilla")
    ap.add_argument("--saltar-preprocessor", action="store_true",
                    help="no correr los pasos del preprocesador (ya existen los Parquet)")
    args = ap.parse_args()

    # Convierte el texto '--etapas a,b' en una lista y revisa que las etapas existan
    etapas = [e.strip() for e in args.etapas.split(",") if e.strip()]
    malas = [e for e in etapas if e not in ALL]
    if malas:
        sys.exit(f"Etapas desconocidas: {malas}. Opciones: {ALL}")
    if args.saltar_preprocessor and "preprocesar" in etapas:
        etapas.remove("preprocesar")
    cfg = ["--config", args.config] if args.config else []

    sys.path.insert(0, str(SRC))
    from common import load_config
    cfg_dict = load_config(args.config)

    py = sys.executable
    # Cada etapa ejecuta su propio script
    for e in etapas:
        print(f"\n{'=' * 70}\n ETAPA: {e}\n{'=' * 70}", flush=True)
        if e == "preprocesar":
            etapa_preprocesar(cfg_dict)
        elif e == "splits":
            run([py, SRC / "1_prepare_splits.py", *cfg], ROOT)
        elif e == "diagnostico":
            run([py, SRC / "2_diagnose.py", *cfg], ROOT)
        elif e == "optimizar":
            run([py, SRC / "3_optimize.py", "--all", *(["--fast"] if args.rapido else []), *cfg], ROOT)
        elif e == "borde":
            run([py, SRC / "4_edge_benchmark.py", *cfg], ROOT)
        elif e == "evaluar":
            run([py, SRC / "5_evaluate.py", *cfg], ROOT)
        elif e == "reporte":
            run([py, SRC / "6_report.py", *cfg], ROOT)
    print("\nListo.")


if __name__ == "__main__":
    main()
