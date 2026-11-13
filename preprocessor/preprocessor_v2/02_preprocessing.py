import pandas as pd
import numpy as np
import os
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import gc
from config import load_config

def calculate_global_stats(parquet_path, info, chunksize):
    """Calcula stats sobre el dataset de TRAIN para evitar leakage."""
    num_cols_list = info.get('numerical_cols', [])
    if not num_cols_list:
        return None, None, None
        
    global_mins = pd.Series(dtype='float64')
    global_maxs = pd.Series(dtype='float64')
    global_sums = pd.Series(dtype='float64')
    global_counts = pd.Series(dtype='float64')
    
    parquet_file = pq.ParquetFile(parquet_path)
    for batch in parquet_file.iter_batches(batch_size=chunksize):
        chunk = batch.to_pandas()
        num_cols_in_chunk = [c for c in num_cols_list if c in chunk.columns]
        for col in num_cols_in_chunk:
            chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
        chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
            
        global_mins = chunk[num_cols_in_chunk].min().combine(global_mins, min)
        global_maxs = chunk[num_cols_in_chunk].max().combine(global_maxs, max)
        global_sums = chunk[num_cols_in_chunk].sum().add(global_sums, fill_value=0)
        global_counts = chunk[num_cols_in_chunk].count().add(global_counts, fill_value=0)
        
    global_means = global_sums / global_counts
    return global_mins, global_maxs, global_means

def process_chunk(chunk, info, imputer, scaler, all_encoded_columns=None):
    num_cols = [c for c in info.get('numerical_cols', []) if c in chunk.columns]
    target_col = info['label']
    sub_category_col = info['sub_category_col']
    category_col = info['category_col']

    for col in num_cols:
         chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
    chunk.replace([np.inf, -np.inf], np.nan, inplace=True)

    if num_cols and imputer and scaler:
        chunk[num_cols] = imputer.transform(chunk[num_cols])
        chunk[num_cols] = scaler.transform(chunk[num_cols])

    chunk.dropna(subset=[target_col, sub_category_col, category_col], inplace=True)

    target = chunk.pop(target_col)
    sub_category_data = chunk.pop(sub_category_col)
    category_data = chunk.pop(category_col)

    target = target.astype(str).map({str(info['normal_label']): 0, str(info['anomaly_label']): 1}).astype('int8')

    chunk.drop(columns=info.get('id_cols', []), errors='ignore', inplace=True)

    cat_cols = [c for c in info.get('categorical_cols', []) if c in chunk.columns and c != category_col]
    if cat_cols:
        chunk = pd.get_dummies(chunk, columns=cat_cols, dtype=np.int8)
        if all_encoded_columns:
            chunk = chunk.reindex(columns=all_encoded_columns, fill_value=0)

    chunk[target_col] = target
    chunk[sub_category_col] = sub_category_data
    chunk[category_col] = category_data

    return chunk

def build_schema(sample_df, info):
    """Establece las clases que determinan el entorno One-Hot y de tipos Parquet."""
    sample_for_schema = sample_df.drop(columns=[info['label'], info['sub_category_col'], info['category_col']] + info.get('id_cols', []), errors='ignore')
    cat_cols_for_fitting = [c for c in info.get('categorical_cols', []) if c in sample_for_schema.columns and c != info['category_col']]
    encoded_sample = pd.get_dummies(sample_for_schema, columns=cat_cols_for_fitting, dtype=np.int8)
    
    final_feature_columns = list(encoded_sample.columns)
    
    temp_arrow = pa.Table.from_pandas(encoded_sample, preserve_index=False)
    final_schema_fields = []
    for field in temp_arrow.schema:
        if field.name in info.get('numerical_cols', []):
            final_schema_fields.append(pa.field(field.name, pa.float64()))
        else:
            final_schema_fields.append(field)
            
    final_schema_fields.append(pa.field(info['label'], pa.int8()))
    final_schema_fields.append(pa.field(info['sub_category_col'], pa.string()))
    final_schema_fields.append(pa.field(info['category_col'], pa.string()))
    
    return pa.schema(final_schema_fields), final_feature_columns

def transform_and_save(input_path, output_path, info, imputer, scaler, schema, final_features, chunksize):
    writer = None
    if not os.path.exists(input_path):
        return
        
    parquet_file = pq.ParquetFile(input_path)
    for batch in parquet_file.iter_batches(batch_size=chunksize):
        chunk = batch.to_pandas()
        processed = process_chunk(chunk, info, imputer, scaler, final_features)
        
        final_output_order = [field.name for field in schema]
        processed = processed.reindex(columns=final_output_order, fill_value=0)
        
        for field in schema:
            if field.name in processed.columns:
                if pa.types.is_integer(field.type) or pa.types.is_floating(field.type):
                    processed[field.name] = processed[field.name].astype(str(field.type), errors='ignore')
                elif pa.types.is_string(field.type):
                    processed[field.name] = processed[field.name].astype(str).replace('nan', '', regex=False)
                    
        table = pa.Table.from_pandas(processed, schema=schema, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_path, schema)
        writer.write_table(table)
        
    if writer:
        writer.close()

def main(override_datasets=None):
    conf = load_config()
    if not conf: return
    
    ROOT = conf['ROOT']
    CHUNKSIZE = conf['CHUNKSIZE']
    DATASETS = conf['DATASETS']
    datasets_to_process = override_datasets or conf['DATASETS_TO_PROCESS']
    base_parquet_path = os.path.join(ROOT, "Datasets", "Parquet")

    for name in datasets_to_process:
        if name not in DATASETS: continue
        info = DATASETS[name]
        
        dataset_dir = os.path.join(base_parquet_path, name)
        train_raw = os.path.join(dataset_dir, "train_raw.parquet")
        test_raw = os.path.join(dataset_dir, "test_raw.parquet")
        
        if not os.path.exists(train_raw) or not os.path.exists(test_raw):
            print(f"Skipping {name}, train_raw/test_raw no existen.")
            continue
            
        print(f"\n{'='*40}\n⚙️  Fase 2: Preprocesando Dataset (No Leakage): {name}\n{'='*40}")
        
        # 1. Aprender estadísticas SOLO en TRAIN
        print("Calculando estadísticas base evaluando únicamente el subset TRAIN...")
        mins, maxs, means = calculate_global_stats(train_raw, info, CHUNKSIZE)
        
        imputer, scaler = None, None
        num_cols = info.get('numerical_cols', [])
        if mins is not None and not mins.empty:
            imputer = SimpleImputer(strategy='mean')
            dummy_imputer = pd.DataFrame([means], columns=means.index)
            imputer.fit(dummy_imputer[num_cols])
            
            scaler = MinMaxScaler()
            dummy_scaler = pd.DataFrame([mins, maxs], columns=mins.index)
            scaler.fit(dummy_scaler[num_cols])
        
        # 2. Definir Schema OHE utilizando el propio train
        print("Definiendo el esquema Parquet final...")
        sample_df = pq.read_table(train_raw).slice(0, min(CHUNKSIZE, 10000)).to_pandas()
        schema, final_feats = build_schema(sample_df, info)

        # 3. Process Train
        train_processed = os.path.join(dataset_dir, "train_processed.parquet")
        print("Procesando TRAIN...")
        transform_and_save(train_raw, train_processed, info, imputer, scaler, schema, final_feats, CHUNKSIZE)
        
        # 4. Process Test
        test_processed = os.path.join(dataset_dir, "test_processed.parquet")
        print("Procesando TEST...")
        transform_and_save(test_raw, test_processed, info, imputer, scaler, schema, final_feats, CHUNKSIZE)
        
        print(f"✅ Preprocesamiento exitoso para {name}.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Paso 2: Preprocesamiento (sin data leakage)')
    parser.add_argument('datasets', nargs='*', help='Datasets a procesar (default: todos los de config.yaml)')
    args = parser.parse_args()
    main(override_datasets=args.datasets or None)
