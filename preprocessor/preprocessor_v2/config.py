import yaml
import json
from pathlib import Path
import os

def load_config(yaml_path='../config.yaml', json_path='../config.json'):
    """
    Carga las configuraciones de config.yaml y config.json ubicados 
    usualmente en la carpeta padre de `preprocessor_v2`.
    Retorna un diccionario unificado con los parámetros indispensables.
    """
    # Intentar también rutas absolutas de ejecución si corre desde otra carpeta
    if not os.path.exists(yaml_path):
        yaml_path = 'config.yaml'
    if not os.path.exists(json_path):
        json_path = 'config.json'

    try:
        with open(yaml_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        env = yaml_config.get('environment_current', 'default')
        if env not in yaml_config.get('environments', {}):
            raise KeyError(f"Entorno configurado '{env}' no encontrado en {yaml_path}")
            
        env_dict = yaml_config['environments'][env]
        root_path = Path(env_dict.get('ROOT', '.'))
        chunksize = env_dict.get('CHUNKSIZE', 500000)
        datasets_to_process = env_dict.get('datasets_to_process', [])
        num_features = env_dict.get('NUM_FEATURES', 40)
        
        with open(json_path, 'r') as f:
            dataset_configs = json.load(f)
            
        print(f"✅ Configuración cargada con éxito (Entorno: {env})")
        return {
            'ROOT': root_path,
            'CHUNKSIZE': chunksize,
            'DATASETS_TO_PROCESS': datasets_to_process,
            'NUM_FEATURES': num_features,
            'DATASETS': dataset_configs
        }
    except Exception as e:
        print(f"❌ Error crítico inicializando configuraciones: {e}")
        return None
