"""
Preprocesador de fuente para el dataset DS2OS.

Transformaciones específicas:
  - La columna 'normality' contiene tanto el label como la subcategoría
    en formato 'anomalous(tipo_ataque)'. Se separa en:
      * label        → 'normal' | 'anomalous'
      * category     → 'normal' | 'anomalous'
      * sub_category → 'normal' | tipo específico de ataque
  - La columna 'value' se castea a numérico (puede contener texto).
"""

from __future__ import annotations

import re
from typing import Iterator

import pandas as pd

from base_preprocessor import SourcePreprocessor


class DS2OSPreprocessor(SourcePreprocessor):

    _ATTACK_RE = re.compile(r"^anomalous\((.+)\)$")

    def _iter_raw_chunks(self) -> Iterator[pd.DataFrame]:
        # DS2OS es pequeño, se carga completo para evitar inconsistencias
        # al parsear la columna 'normality' con chunking.
        df = self._read_full_csv(
            self.output_path.parent / "mainSimulationAccessTraces.csv"
        )
        yield df

    def transform_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        df = chunk.copy()

        # --- Crear columnas estándar desde 'normality' ---
        df["sub_category"] = df["normality"].apply(self._extract_sub_category)
        df["category"] = df["normality"].apply(
            lambda x: "normal" if x == "normal" else "anomalous"
        )
        df["normality"] = df["category"]  # label binario

        # --- Cast numérico mínimo ---
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        return df

    def _extract_sub_category(self, value: str) -> str:
        if value == "normal":
            return "normal"
        match = self._ATTACK_RE.match(str(value))
        return match.group(1) if match else str(value)
