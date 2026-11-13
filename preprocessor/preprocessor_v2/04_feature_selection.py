import pandas as pd
import numpy as np
import os
import gc
from config import load_config

from sklearn.feature_selection import chi2, f_classif, mutual_info_classif, SelectKBest, SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Selectores a ejecutar
SELECTORS_TO_RUN = ['vif', 'f_classif', 'mutual_info_classif', 'lgbm_importance', 'l1_lasso', 'chi2']

# ── Nota sobre el uso de la muestra estratificada ─────────────────────────
# Para datasets grandes, la selección de características se hace sobre el
# subconjunto estratificado (train_stratified) generado en el paso 03.
# Justificación por método:
#   - VIF: mide estructura de correlación; estable con >10K filas representativas.
#   - Chi2/F/MI: rank order de features estable en muestras estratificadas grandes.
#   - LightGBM: usa subsample nativo; se entrena sobre la muestra para consistencia.
#   - L1 Lasso: se usa SGDClassifier con partial_fit para entrenamiento incremental.
# La muestra estratificada garantiza representación proporcional de todas las
# combinaciones (category, sub_category), por lo que es estadísticamente válida.
# ──────────────────────────────────────────────────────────────────────────── 
DEFAULT_USE_SAMPLE = True  # Usar train_stratified por defecto (override en config.json)

def calculate_vif(df_numeric):
    vif_data = pd.DataFrame()
    vif_data["feature"] = df_numeric.columns
    # Tomamos una pequeña muestra si el df es inmenso para hacer VIF manejable computacionalmente.
    df_eval = df_numeric.sample(min(10000, len(df_numeric)), random_state=42)
    vif_data["VIF"] = [variance_inflation_factor(df_eval.values, i) for i in range(df_eval.shape[1])]
    return vif_data.sort_values(by="VIF", ascending=False)

def reduce_features_with_vif(df_numeric, vif_threshold=5.0):
    columns_to_process = df_numeric.columns.tolist()
    columns_removed = []
    
    while True:
        df_subset = df_numeric[columns_to_process]
        if df_subset.shape[1] < 2: break
            
        vif_data = calculate_vif(df_subset)
        highest_vif = vif_data.iloc[0]
        
        if highest_vif["VIF"] > vif_threshold:
            columns_to_process.remove(highest_vif["feature"])
            columns_removed.append(highest_vif["feature"])
        else:
            break
            
    print(f"   -> Se eliminaron {len(columns_removed)} características multicolineales vía VIF.")
    return df_numeric[columns_to_process], columns_removed


def main(override_datasets=None):
    conf = load_config()
    if not conf: return
    
    ROOT = conf['ROOT']
    DATASETS = conf['DATASETS']
    NUM_FEATURES = conf['NUM_FEATURES']
    datasets_to_process = override_datasets or conf['DATASETS_TO_PROCESS']
    base_parquet_path = os.path.join(ROOT, "Datasets", "Parquet")

    univariate_selectors = {"chi2": chi2, "f_classif": f_classif, "mutual_info_classif": mutual_info_classif}

    for name in datasets_to_process:
        if name not in DATASETS: continue
        info = DATASETS[name]
        label_col = info['label']
        
        print(f"\n{'='*50}\n🎯 Fase 4: Selección de Características: {name}\n{'='*50}")
        dataset_dir = os.path.join(base_parquet_path, name)
        
        # Usar muestra estratificada por defecto, o completo si se configura explícitamente
        use_sample = bool(info.get('use_sample_for_selection', DEFAULT_USE_SAMPLE))
        file_suffix = "stratified" if use_sample else "processed"
        train_path = os.path.join(dataset_dir, f"train_{file_suffix}.parquet")
        test_path = os.path.join(dataset_dir, f"test_{file_suffix}.parquet")
        
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            print(f"❌ Archivos {file_suffix} no encontrados para {name}. Ejecuta el paso previo primero.")
            continue
            
        print(f"Cargando datasets base para selección ({file_suffix})...")
        df_train = pd.read_parquet(train_path)
        df_test  = pd.read_parquet(test_path)
        print(f"-> Train: {len(df_train):,} filas | Test: {len(df_test):,} filas")
        
        y_train = df_train[label_col]
        X_train = df_train.drop(columns=[label_col], errors='ignore')
        
        numerical_cols = X_train.select_dtypes(include=np.number).columns.tolist()
        metadata_cols = [col for col in [info.get('category_col'), info.get('sub_category_col')] if col in X_train.columns]
        
        X_train_numeric = X_train[numerical_cols]
        print(f"-> Base: {X_train_numeric.shape[1]} características numéricas disponibles.")

        for selector_name in SELECTORS_TO_RUN:
            print(f"\n--- Aplicando {selector_name.upper()} ---")
            
            selector_dir = os.path.join(dataset_dir, selector_name)
            train_out_dir = os.path.join(selector_dir, "train")
            test_out_dir = os.path.join(selector_dir, "test")
            
            # Tolerancia a Fallos/Interrupciones: Evitar recalcular si el archivo ya existe
            final_artifact = os.path.join(test_out_dir, "best_features.parquet")
            if os.path.exists(final_artifact):
                print("   ⏩ [SALTADO - SELECCIÓN PREVIAMENTE CALCULADA]")
                continue
            
            # Se usa una copia exacta para evitar manipulaciones cruzadas en el loop
            df_train_num_copy = X_train_numeric.copy()
            selected_cols = []
            
            try:
                if selector_name == 'vif':
                    # VIF no depende de 'k', se basa en el threshold y correlación iterativa.
                    print("   -> Iniciando cálculo VIF...")
                    df_clean, _ = reduce_features_with_vif(df_train_num_copy)
                    selected_cols = df_clean.columns.tolist()
                    print("   -> Cálculo VIF finalizado.")

                else:
                    k = min(NUM_FEATURES, X_train_numeric.shape[1])
                    if k <= 0: continue

                    if selector_name in univariate_selectors:
                        print(f"   -> Iniciando SelectKBest con {selector_name}...")
                        selector_model = SelectKBest(univariate_selectors[selector_name], k=k)
                        # chi2 requiere datos no negativos (MinMaxScaler ya lo ha hecho en el Paso 2)
                        selector_model.fit(df_train_num_copy, y_train)
                        selected_cols = df_train_num_copy.columns[selector_model.get_support()].tolist()
                        print(f"   -> {selector_name.upper()} finalizado.")

                    elif selector_name == 'lgbm_importance':
                        # LightGBM tiene subsampling nativo; no requiere cargar todo en RAM
                        print("   -> Iniciando LGBMClassifier...")
                        model = LGBMClassifier(
                            random_state=42,
                            n_jobs=-1,
                            verbose=-1,
                            subsample=0.8,          # usa 80% de las filas por árbol
                            subsample_freq=1,
                            colsample_bytree=0.8,   # usa 80% de features por árbol
                        )
                        model.fit(df_train_num_copy, y_train)
                        importances = pd.Series(model.feature_importances_, index=df_train_num_copy.columns)
                        selected_cols = importances.nlargest(k).index.tolist()
                        print("   -> LGBM finalizado.")

                    elif selector_name == 'l1_lasso':
                        # Para Lasso escalamos primero temporalmente interno
                        print("   -> Iniciando Lasso (LogReg L1)...")
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(df_train_num_copy)
                        model = LogisticRegression(penalty='l1', C=0.1, solver='liblinear', random_state=42, max_iter=500)
                        selector_model = SelectFromModel(model, max_features=k, prefit=False)
                        selector_model.fit(X_train_scaled, y_train)
                        selected_cols = df_train_num_copy.columns[selector_model.get_support()].tolist()
                        print("   -> Lasso finalizado.")
                
                print(f"-> {len(selected_cols)} variables seleccionadas.")
                
                final_keep = list(set(metadata_cols + [label_col] + selected_cols))
                
                df_train_best = df_train[[c for c in final_keep if c in df_train.columns]]
                df_test_best = df_test[[c for c in final_keep if c in df_test.columns]]
                
                os.makedirs(train_out_dir, exist_ok=True)
                os.makedirs(test_out_dir, exist_ok=True)
                
                df_train_best.to_parquet(os.path.join(train_out_dir, "best_features.parquet"), index=False)
                df_test_best.to_parquet(os.path.join(test_out_dir, "best_features.parquet"), index=False)
                
            except Exception as e:
                print(f"❌ Error ejecutando selector {selector_name}: {e}")
                
        # Limpieza pesada post-iteración
        del df_train, df_test, X_train, y_train, X_train_numeric
        gc.collect()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Paso 4: Selección de características')
    parser.add_argument('datasets', nargs='*', help='Datasets a procesar (default: todos los de config.yaml)')
    args = parser.parse_args()
    main(override_datasets=args.datasets or None)
