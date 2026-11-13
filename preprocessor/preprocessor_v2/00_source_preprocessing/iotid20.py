"""
Preprocesador de fuente para IoTID20.

Transformaciones específicas:
  - El archivo CSV es único y grande (~1GB).
  - Las columnas ya tienen el esquema correcto:
      label='Label', category_col='Cat', sub_category_col='Sub_Cat'
  - Las columnas 'id_cols' (Timestamp, Flow_ID, IPs, puertos) se mantienen
    intactas aquí; serán eliminadas en 02_preprocessing.py.
  - El Timestamp ya viene como string formateado; no se convierte aquí.
"""

from __future__ import annotations

from typing import Iterator

import pandas as pd

from base_preprocessor import SourcePreprocessor


class IoTID20Preprocessor(SourcePreprocessor):

    def _iter_raw_chunks(self) -> Iterator[pd.DataFrame]:
        source_path = (
            self.output_path.parent / "IoTNetworkIntrusionDataset.csv"
        )
        yield from self._read_csv_chunks(source_path)

    def transform_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """
        IoTID20 ya tiene el esquema estándar del proyecto.
        Solo se limpian infinitos y se aseguran los tipos numéricos básicos.
        """
        import numpy as np

        df = chunk.copy()
        num_cols = [c for c in self.info.get("numerical_cols", []) if c in df.columns]
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.replace([float("inf"), float("-inf")], float("nan"), inplace=True)

        return df
