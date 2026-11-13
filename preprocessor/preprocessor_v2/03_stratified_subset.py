"""
Paso 3: Muestreo estratificado configurable por dataset.

La fracción de muestreo se lee desde la configuración del dataset en config.json
mediante el campo 'sample_fraction'. Si no está definido, el default es 0.10 (10%).

Esto permite que datasets gigantes como IOT23 tengan variantes con distintas
fracciones (10%, 1%) sin cambiar el código:
  - IOT23          → sample_fraction: 1.0   (completo, sin muestreo)
  - IOT23-10percent → sample_fraction: 0.10
  - IOT23-1percent  → sample_fraction: 0.01

La estratificación usa la clave compuesta (category_col, sub_category_col)
para garantizar que subcategorías con el mismo nombre en distintas categorías
se traten como grupos separados.
"""

import pandas as pd
import os
import gc
from config import load_config

DEFAULT_FRACTION = 0.10
RANDOM_STATE = 42


def resample_and_save(
    input_path: str,
    output_path: str,
    category_col: str | None,
    sub_category_col: str | None,
    fraction: float,
) -> None:
    """
    Carga un parquet, realiza el muestreo estratificado con la fracción dada
    y guarda el resultado en output_path.

    Args:
        input_path:       Ruta al archivo Parquet de entrada.
        output_path:      Ruta donde se guardará la muestra.
        category_col:     Columna de categoría (primer nivel de estratificación).
        sub_category_col: Columna de subcategoría (segundo nivel).
        fraction:         Fracción de muestreo [0.0 – 1.0].
    """
    if not os.path.exists(input_path):
        print(f"❌ Archivo no encontrado: {input_path}")
        return

    print(f"Cargando dataset: {input_path}")
    df = pd.read_parquet(input_path)

    # Fracción 1.0 → copiar sin muestrear (dataset completo)
    if fraction >= 1.0:
        print(f"-> Fracción 1.0: copiando dataset completo ({len(df):,} filas).")
        df.to_parquet(output_path, index=False)
        del df
        gc.collect()
        return

    # Construir clave compuesta (category, sub_category) para estratificación
    strat_cols = [c for c in [category_col, sub_category_col] if c and c in df.columns]

    if not strat_cols:
        print("⚠️  Columnas de estratificación no encontradas. Fallback a muestra aleatoria.")
        sampled_df = df.sample(frac=fraction, random_state=RANDOM_STATE)
    else:
        label = " + ".join(strat_cols)
        print(f"Estratificando ({fraction*100:.1f}%) por clave compuesta: [{label}]")

        # Para grupos demasiado pequeños para muestrear (< 1/fraction filas),
        # se incluyen completos para no perder clases minoritarias.
        min_group_size = max(1, int(1 / fraction))
        sampled_df = df.groupby(strat_cols, group_keys=False).apply(
            lambda x: (
                x.sample(frac=fraction, random_state=RANDOM_STATE)
                if len(x) > min_group_size
                else x
            ),
            include_groups=False,
        )
        # groupby con include_groups=False excluye las columnas de agrupación del
        # DataFrame pasado al lambda, pero las restablece en el resultado final.
        # Re-incorporamos las columnas de estratificación que pueden haberse perdido.
        for col in strat_cols:
            if col not in sampled_df.columns:
                sampled_df[col] = df.loc[sampled_df.index, col]

    print(f"-> Muestreo: {len(df):,} → {len(sampled_df):,} filas ({fraction*100:.1f}%).")
    sampled_df.to_parquet(output_path, index=False)

    del df, sampled_df
    gc.collect()


def main(override_datasets=None) -> None:
    conf = load_config()
    if not conf:
        return

    ROOT = conf["ROOT"]
    DATASETS = conf["DATASETS"]
    datasets_to_process = override_datasets or conf["DATASETS_TO_PROCESS"]
    base_parquet_path = os.path.join(ROOT, "Datasets", "Parquet")

    for name in datasets_to_process:
        if name not in DATASETS:
            continue

        info = DATASETS[name]
        category_col = info.get("category_col")
        sub_category_col = info.get("sub_category_col")

        # Fracción configurable por dataset en config.json
        fraction = float(info.get("sample_fraction", DEFAULT_FRACTION))

        print(f"\n{'='*50}")
        print(f"🤏 Fase 3: Submuestreo Estratificado ({fraction*100:.1f}%): {name}")
        print(f"{'='*50}")

        dataset_dir = os.path.join(base_parquet_path, name)
        train_path = os.path.join(dataset_dir, "train_processed.parquet")
        test_path = os.path.join(dataset_dir, "test_processed.parquet")
        train_out = os.path.join(dataset_dir, "train_stratified.parquet")
        test_out = os.path.join(dataset_dir, "test_stratified.parquet")

        print("Muestreando conjunto TRAIN...")
        resample_and_save(train_path, train_out, category_col, sub_category_col, fraction)

        print("Muestreando conjunto TEST...")
        resample_and_save(test_path, test_out, category_col, sub_category_col, fraction)

        print(f"✅ Muestreo completado para {name}.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Paso 3: Submuestreo estratificado')
    parser.add_argument('datasets', nargs='*', help='Datasets a procesar (default: todos los de config.yaml)')
    args = parser.parse_args()
    main(override_datasets=args.datasets or None)
