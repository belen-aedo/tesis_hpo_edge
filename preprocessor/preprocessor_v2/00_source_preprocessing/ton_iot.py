"""
Preprocesador de fuente para TON-IoT Network Dataset.

Transformaciones específicas:
  - El dataset tiene el label de ataque en 'category' (normal / tipo_ataque)
    y el tipo específico en 'label' (subcategoría).
  - Las columnas del esquema del proyecto requieren mapeo al estándar:
      * info['label']            = 'category'  (normal / anormal)
      * info['category_col']     = 'label'     (categoría de ataque)
      * info['sub_category_col'] = 'type'      (subcategoría)
  - Se crea la columna binaria estándar 'category' reemplazando los
    valores de texto por 'normal' / 'anormal'.
"""

from __future__ import annotations

from typing import Iterator

import pandas as pd

from base_preprocessor import SourcePreprocessor


class TonIoTPreprocessor(SourcePreprocessor):

    def _iter_raw_chunks(self) -> Iterator[pd.DataFrame]:
        source_path = (
            self.output_path.parent / "Train_Test_Labels.csv"
        )
        if not source_path.exists():
            source_path = self.output_path
        yield from self._read_csv_chunks(source_path)

    def transform_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        df = chunk.copy()

        # En TON-IoT, la columna 'label' trae 0/1 y 'type' el tipo de ataque.
        # Se genera la columna 'category' con texto normal/anormal para el label binario.
        normal_lbl  = self.info["normal_label"]   # 'normal'
        anomaly_lbl = self.info["anomaly_label"]  # 'anormal'

        if "label" in df.columns:
            df["category"] = df["label"].apply(
                lambda x: normal_lbl if str(x) in ("0", "normal") else anomaly_lbl
            )

        return df
