"""
Preprocesador de fuente para IOT23.

El dataset IOT23 consiste en múltiples capturas de tráfico Zeek/Bro
almacenadas como archivos '.labeled' dispersos dentro de una estructura
de directorios bajo ROOT/Datasets/Sources/IOT23/opt/.

Transformaciones específicas:
  - Recursión por directorios buscando archivos con extensión '.labeled'.
  - Parseo con separador mixto (espacios y/o tabulaciones) y nombres de
    columna fijos (esquema Zeek conn.log estándar).
  - El símbolo "-" (valor ausente de Zeek) se reemplaza por NaN.
  - Se crea la columna 'category' (binaria normal/malicioso) desde 'label'.
  - Para filas benignas, 'detailed-label' se establece como 'benign'.
  - La columna 'detailed-label' actúa como sub_category_col.

Nota: la generación del subconjunto 10% es responsabilidad de
03_stratified_subset.py; este script solo produce preprocessed.csv.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from base_preprocessor import SourcePreprocessor

# Esquema de columnas fijo del formato Zeek conn.log .labeled
_ZEEK_COLUMNS = [
    "ts", "uid", "id.orig_h", "id.orig_p",
    "id.resp_h", "id.resp_p", "proto", "service",
    "duration", "orig_bytes", "resp_bytes", "conn_state",
    "local_orig", "local_resp", "missed_bytes", "history",
    "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes",
    "tunnel_parents", "label", "detailed-label",
]


class IOT23Preprocessor(SourcePreprocessor):

    def _iter_raw_chunks(self) -> Iterator[pd.DataFrame]:
        """
        Recorre recursivamente el directorio de fuente buscando archivos
        '.labeled' y los entrega en chunks para no saturar la memoria RAM.
        El dataset IOT23 puede superar los 20 GB en disco.
        """
        source_dir = Path(self.root) / "Datasets" / "Sources" / "IOT23" / "opt"

        labeled_files = sorted(source_dir.rglob("*.labeled"))
        if not labeled_files:
            raise FileNotFoundError(
                f"No se encontraron archivos '.labeled' en {source_dir}"
            )

        for file_path in labeled_files:
            try:
                for chunk in pd.read_csv(
                    file_path,
                    names=_ZEEK_COLUMNS,
                    comment="#",                    # Zeek escribe cabeceras con '#'
                    sep=r"\s+|\t+",                 # separador mixto espacio/tab
                    engine="python",
                    chunksize=self.chunksize,
                    on_bad_lines="skip",            # tolerancia a líneas malformadas
                ):
                    yield chunk
            except Exception as e:
                # Si un archivo está corrupto, se registra y se continúa
                import logging
                logging.getLogger(__name__).warning(
                    "Error leyendo %s: %s – se omite.", file_path, e
                )
                continue

    def transform_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        df = chunk.copy()

        # 1. Zeek usa "-" para indicar ausencia de valor → NaN
        df.replace("-", np.nan, inplace=True)

        # 2. Normalizar label a minúsculas para comparaciones robustas
        #    (el notebook encontró tanto 'bening' como 'benign' por typo)
        df["label"] = df["label"].astype(str).str.lower().str.strip()

        # 3. Crear columna 'category' binaria: normal | malicious
        benign_aliases = {"benign", "bening"}   # 'bening' es un typo frecuente en IOT23
        df["category"] = df["label"].apply(
            lambda x: self.info["normal_label"]   # 'benign'
            if x in benign_aliases
            else self.info["anomaly_label"]        # 'malicious'
        )

        # 4. Para tráfico benigno, detailed-label debe coincidir con 'benign'
        df.loc[df["label"].isin(benign_aliases), "detailed-label"] = "benign"

        return df
