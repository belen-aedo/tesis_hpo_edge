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
        
        chunk_iter = pd.read_csv(input_csv_path, chunksize=CHUNKSIZE, low_memory=False)
        
        for i, chunk in enumerate(chunk_iter):
            print(f"-> Separando chunk {i+1}...")
            chunk = enforce_types(chunk, info)
            
            # Sub-división por cada chunk (80%/20%), aleatorio.
            # Dada la ley de grandes números, split en chunks de 500k preserva la distribución de clases.
            train_chunk, test_chunk = train_test_split(chunk, test_size=0.20, random_state=42)
            
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
            
        print(f"✅ Split exitoso guardado en {dataset_output_dir} (train_raw.parquet, test_raw.parquet)")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Paso 1: Train/Test Split por chunks')
    parser.add_argument('datasets', nargs='*', help='Datasets a procesar (default: todos los de config.yaml)')
    args = parser.parse_args()
    main(override_datasets=args.datasets or None)
