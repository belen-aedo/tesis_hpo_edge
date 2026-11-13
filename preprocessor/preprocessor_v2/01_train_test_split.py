import pandas as pd
import numpy as np
import os
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
import gc
from config import load_config

def enforce_types(df, info):
    """
    Fuerza los tipos numéricos y categóricos dictados por la configuración
    para evitar inconsistencias entre chunks.
    """
    num_cols_list = info.get('numerical_cols', [])
    num_cols_in_df = [c for c in num_cols_list if c in df.columns]
    
    for col in num_cols_in_df:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    for col in df.columns:
        if col not in num_cols_in_df:
            # Forzar strings para todo lo que no sea numerico (incluyendo clases y NaNs en categóricas)
            df[col] = df[col].astype(str).replace('nan', '', regex=False)
            
    return df


def _label_cols(info):
    """Columnas de etiqueta: nunca forman parte de la huella de una fila."""
    return [c for c in (info.get('label'), info.get('category_col'), info.get('sub_category_col')) if c]


def _row_hashes(df, info):
    """Huella de cada fila usando solo sus variables (sin identificadores ni etiquetas).
    Dos flujos con la misma huella son el mismo flujo para el modelo."""
    skip = set(info.get('id_cols', [])) | set(_label_cols(info))
    cols = [c for c in df.columns if c not in skip]
    return pd.util.hash_pandas_object(df[cols], index=False).to_numpy()


def find_ambiguous(input_csv_path, info, chunksize):
    """Primera pasada: huellas que aparecen con etiquetas binarias distintas.
    Esas filas son ambiguas (el mismo flujo marcado como normal y como ataque)."""
    target = info['label']
    labels_by_hash = {}
    for chunk in pd.read_csv(input_csv_path, chunksize=chunksize, low_memory=False):
        chunk = enforce_types(chunk, info)
        for h, y in zip(_row_hashes(chunk, info), chunk[target].astype(str)):
            labels_by_hash.setdefault(h, set()).add(y)
    return {h for h, ys in labels_by_hash.items() if len(ys) > 1}


def deduplicate(chunk, info, seen, ambiguous):
    """Quita filas ambiguas y repetidas (conserva la primera aparicion de cada flujo).
    `seen` se comparte entre chunks, asi la deduplicacion es global."""
    h = _row_hashes(chunk, info)
    keep = np.zeros(len(chunk), dtype=bool)
    for i, hi in enumerate(h):
        if hi in ambiguous or hi in seen:
            continue
        seen.add(hi)
        keep[i] = True
    return chunk.loc[keep]


def split_chunk(chunk, info, test_size, seed):
    """Division train/test. Con 'stratify_split' en config.json se estratifica por la
    subcategoria (tipo de ataque); si algun tipo tiene menos de 2 filas en el chunk,
    se divide al azar."""
    strat = None
    col = info.get('sub_category_col')
    if info.get('stratify_split') and col in chunk.columns:
        counts = chunk[col].value_counts()
        if len(counts) > 1 and counts.min() >= 2:
            strat = chunk[col]
    return train_test_split(chunk, test_size=test_size, random_state=seed, stratify=strat)


def main(override_datasets: list[str] | None = None):
    conf = load_config()
    if not conf:
        return
    
    ROOT = conf['ROOT']
    CHUNKSIZE = conf['CHUNKSIZE']
    DATASETS = conf['DATASETS']
    datasets_to_process = override_datasets or conf['DATASETS_TO_PROCESS']
    base_parquet_path = os.path.join(ROOT, "Datasets", "Parquet")

    for name in datasets_to_process:
        if name not in DATASETS:
            print(f"⚠️  Dataset {name} no listado en config.json. Omitiendo.")
            continue
            
        info = DATASETS[name]
        print(f"\n{'='*40}\n🚂 Fase 1: Dividiendo Dataset (Train/Test Split): {name}\n{'='*40}")
        
        input_csv_path = os.path.join(ROOT, info['path'].lstrip(os.sep))
        if not os.path.exists(input_csv_path):
            print(f"❌ Archivo no encontrado: {input_csv_path}")
            continue

        dataset_output_dir = os.path.join(base_parquet_path, name)
        os.makedirs(dataset_output_dir, exist_ok=True)
        train_parquet_file = os.path.join(dataset_output_dir, "train_raw.parquet")
        test_parquet_file = os.path.join(dataset_output_dir, "test_raw.parquet")

        train_writer = None
        test_writer = None
        
        print("Calculando esquema base seguro usando primeros chunks...")
        try:
            sample_df = pd.read_csv(input_csv_path, nrows=CHUNKSIZE, low_memory=False)
            sample_df = enforce_types(sample_df, info)
            schema = pa.Table.from_pandas(sample_df, preserve_index=False).schema
        except Exception as e:
            print(f"Error definiendo el esquema para {name}: {e}")
            continue
        
        # Opciones (config.json); sin ellas el comportamiento es el original
        test_size = float(info.get('test_size', 0.20))
        seed = int(info.get('split_seed', 42))
        dedup = bool(info.get('deduplicate', False))
        seen, ambiguous = set(), set()
        n_in = n_out = 0
        if dedup:
            print("Buscando flujos con etiquetas contradictorias (primera pasada)...")
            ambiguous = find_ambiguous(input_csv_path, info, CHUNKSIZE)
            print(f"   huellas ambiguas: {len(ambiguous)}")

        chunk_iter = pd.read_csv(input_csv_path, chunksize=CHUNKSIZE, low_memory=False)

        for i, chunk in enumerate(chunk_iter):
            print(f"-> Separando chunk {i+1}...")
            chunk = enforce_types(chunk, info)
            n_in += len(chunk)
            # Con 'deduplicate', cada flujo aparece una sola vez: asi una copia no puede
            # quedar en train y otra en test.
            if dedup:
                chunk = deduplicate(chunk, info, seen, ambiguous)
            n_out += len(chunk)

            # Sub-división por cada chunk (por defecto 80%/20%, aleatoria o estratificada).
            train_chunk, test_chunk = split_chunk(chunk, info, test_size, seed)
            
            train_table = pa.Table.from_pandas(train_chunk, schema=schema, preserve_index=False)
            test_table = pa.Table.from_pandas(test_chunk, schema=schema, preserve_index=False)
            
            if train_writer is None:
                train_writer = pq.ParquetWriter(train_parquet_file, schema)
                test_writer = pq.ParquetWriter(test_parquet_file, schema)
                
            train_writer.write_table(train_table)
            test_writer.write_table(test_table)
            
            del chunk, train_chunk, test_chunk, train_table, test_table
            gc.collect()

        if train_writer:
            train_writer.close()
            test_writer.close()
            
        if dedup:
            print(f"   deduplicacion: {n_in:,} filas -> {n_out:,} ({n_in - n_out:,} repetidas o ambiguas eliminadas)")
        print(f"✅ Split exitoso guardado en {dataset_output_dir} (train_raw.parquet, test_raw.parquet)")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Paso 1: Train/Test Split por chunks')
    parser.add_argument('datasets', nargs='*', help='Datasets a procesar (default: todos los de config.yaml)')
    args = parser.parse_args()
    main(override_datasets=args.datasets or None)
