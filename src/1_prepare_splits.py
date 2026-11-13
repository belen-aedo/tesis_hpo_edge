"""
1_prepare_splits.py - Etapa 1: toma la salida del preprocessor y arma
los bloques del experimento (los 3 bloques del diagrama aprobado).

    train : solo trafico normal, sin etiquetas
    val   : solo trafico normal, sin etiquetas (calibra el umbral y calcula el objetivo)
    test  : normal + ataque, con etiquetas y separado del resto. Se divide en dos
            partes sin filas en comun, manteniendo la proporcion de cada tipo de
            ataque y con semilla fija:
              test_oracle : solo la usa el escenario oraculo (maximiza el MCC)
              test_final  : se abre una sola vez (5_evaluate.py) y es la misma
                            para todos los escenarios

De donde sale cada bloque
-------------------------
El preprocessor entrega `train_*` (80 %) y `test_*` (20 %), ambos con trafico
normal y ataques. Su escalado MinMax y su imputacion se ajustan solo con su train.
  * train y val salen del trafico normal del train del preprocessor. Los
    detectores aprenden solo lo normal, asi que sus ataques se descartan.
  * test es el test del preprocessor completo. Asi, los ataques de prueba nunca
    se usaron para ajustar el preprocessor.

El resto de la cadena no usa etiquetas
--------------------------------------
  * Seleccion de caracteristicas (features.mode = unsupervised): usa solo el
    train normal (varianza/frecuencia y correlacion). No mira etiquetas.
  * Reescalado (scaling.method = standard): media y desviacion del train normal.
    Las columnas 0/1 (one-hot) se dejan igual.

USO
    python src/1_prepare_splits.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import (datasets_dir, find_table, io_ext, load_config, outputs_dir, read_table,
                    save_json, set_seeds, sha256_of, splits_dir, write_table)

HINT = {
    "stratified": "paso 03 (03_stratified_subset.py)",
    "processed": "paso 02 (02_preprocessing.py)",
    "selector": "paso 04 (04_feature_selection.py)",
}


# --------------------------------------------------------------------------- #
# Lectura de la salida del preprocessor
# --------------------------------------------------------------------------- #
# Decide que archivos del preprocessor se leen, segun la configuracion
def input_bases(cfg) -> tuple[Path, Path, str]:
    base = datasets_dir(cfg) / "Parquet" / cfg["dataset"]["name"]
    mode = cfg["features"]["mode"]
    if mode == "selector":
        sel = cfg["features"]["selector"]
        return (base / sel / "train" / "best_features", base / sel / "test" / "best_features",
                "selector")
    v = cfg["input"]["variant"]
    if v not in ("stratified", "processed"):
        raise SystemExit(f"input.variant debe ser 'stratified' o 'processed' (es '{v}').")
    return base / f"train_{v}", base / f"test_{v}", v


# Lee el train y el test del preprocessor. Si no existen, explica que paso falta.
def read_inputs(cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr_base, te_base, kind = input_bases(cfg)
    try:
        tr_path, te_path = find_table(tr_base), find_table(te_base)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"\nNo encuentro la salida del preprocessor:\n  {exc}\n"
            f"Falta ejecutar el {HINT[kind]} para '{cfg['dataset']['name']}'.\n"
            f"Carpeta buscada: {tr_base.parent}\n"
            "Revise tambien paths.datasets_root en config/experiment.yaml.")
    print(f"train del preprocessor: {tr_path}\ntest  del preprocessor: {te_path}")
    return read_table(tr_path), read_table(te_path)


def build_target(df: pd.DataFrame, cfg) -> np.ndarray:
    """1 = ataque, 0 = normal."""
    col, nv = cfg["dataset"]["label_col"], cfg["dataset"]["normal_value"]
    if col not in df.columns:
        raise SystemExit(f"La columna de etiqueta '{col}' no esta en los datos. "
                         f"Columnas (primeras 15): {list(df.columns)[:15]}")
    raw = df[col]
    # Si la etiqueta es numerica se compara como numero; si no, como texto
    if pd.api.types.is_numeric_dtype(raw):
        try:
            return (raw.to_numpy(dtype=np.float64) != float(nv)).astype(int)
        except (TypeError, ValueError):
            pass
    return (raw.astype(str).str.strip().str.lower() != str(nv).strip().lower()).astype(int)


# Devuelve el tipo de ataque de cada fila ('normal' si no es ataque)
def attack_types(df: pd.DataFrame, cfg, y: np.ndarray) -> np.ndarray:
    col = cfg["dataset"].get("attack_type_col")
    if col and col in df.columns:
        t = df[col].astype(str).str.strip().to_numpy()
        return np.where(y == 1, t, "normal").astype(object)
    return np.where(y == 1, "ataque", "normal").astype(object)


# Deja solo las variables de entrada: sin etiquetas y solo con columnas numericas
def feature_frame(df: pd.DataFrame, cfg) -> pd.DataFrame:
    drop = set(cfg["dataset"]["drop_cols"]) | {cfg["dataset"]["label_col"]}
    if cfg["dataset"].get("attack_type_col"):
        drop.add(cfg["dataset"]["attack_type_col"])
    X = df.drop(columns=[c for c in df.columns if c in drop])
    X = X.select_dtypes(include=[np.number, "bool"]).astype(np.float32)
    # Los valores infinitos o vacios se reemplazan por 0
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X


# --------------------------------------------------------------------------- #
# Seleccion de caracteristicas y escalado SIN etiquetas (solo train normal)
# --------------------------------------------------------------------------- #
def binary_mask(A: np.ndarray, chunk: int = 256) -> np.ndarray:
    """True en las columnas cuyos valores son solo 0 y 1."""
    out = np.empty(A.shape[1], dtype=bool)
    # Revisa las columnas por grupos para usar menos memoria
    for s in range(0, A.shape[1], chunk):
        blk = A[:, s:s + chunk]
        out[s:s + chunk] = ((blk == 0) | (blk == 1)).all(axis=0)
    return out


def select_unsupervised(Xtr: pd.DataFrame, cfg, seed: int) -> tuple[list[str], dict]:
    """
    Reduce el numero de columnas mirando solo el train normal:
      1. saca las columnas constantes y las columnas 0/1 que casi no cambian (su
         valor menos comun aparece en menos de features.min_minority_freq del
         trafico normal);
      2. las ordena: primero las numericas y despues las 0/1, de mas a menos
         variables (segun p(1-p));
      3. filtro de correlacion: si dos columnas tienen |r| >= max_abs_corr, se
         queda la que va primero. Se detiene al llegar a features.max_features.
    """
    f = cfg["features"]
    A = Xtr.to_numpy(dtype=np.float32)
    cols = np.array(Xtr.columns)
    # is_bin indica que columnas son 0/1
    is_bin = binary_mask(A)
    # Promedio de cada columna. En una columna 0/1, es la proporcion de unos.
    p = A.mean(axis=0)
    # Constante: su valor maximo y minimo son iguales
    constant = A.max(axis=0) == A.min(axis=0)
    # Rara: columna 0/1 cuyo valor menos comun casi no aparece.
    # min(p, 1 - p) es la proporcion de ese valor; se compara con min_minority_freq.
    rare = is_bin & (np.minimum(p, 1 - p) < float(f["min_minority_freq"]))
    # Sobreviven las columnas que no son constantes ni raras
    ok = ~constant & ~rare
    # Separa las candidatas en numericas y binarias
    num_idx = np.flatnonzero(ok & ~is_bin)
    bin_idx = np.flatnonzero(ok & is_bin)
    # Ordena las binarias de mas a menos variables, segun p(1-p)
    bin_idx = bin_idx[np.argsort(-(p[bin_idx] * (1 - p[bin_idx])), kind="stable")]
    # Orden final: numericas primero. Las binarias ocupan los cupos que queden hasta max_features.
    order = np.concatenate([num_idx, bin_idx])
    stats = {"columnas_entrada": int(A.shape[1]), "constantes": int(constant.sum()),
             "binarias_raras": int(rare.sum()), "numericas_candidatas": int(len(num_idx)),
             "binarias_candidatas": int(len(bin_idx))}
    if len(order) == 0:
        raise SystemExit("Tras los filtros no queda ninguna caracteristica: "
                         "revise features.min_minority_freq o los datos de entrada.")

    # Calcula las correlaciones entre columnas con una muestra de hasta 20.000 filas
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(A), size=min(len(A), 20000), replace=False)
    S = A[rows][:, order].astype(np.float64)
    S -= S.mean(axis=0)
    sd = S.std(axis=0)
    sd[sd < 1e-12] = np.inf                       # columna constante en la submuestra: r = 0
    S /= sd
    C = (S.T @ S) / len(S)

    # Recorre las columnas en orden y decide cuales conservar
    kept_pos: list[int] = []
    n_corr = 0
    for i in range(len(order)):
        # Se detiene al llegar al tope de columnas
        if len(kept_pos) >= int(f["max_features"]):
            break
        # Descarta la columna si esta muy correlacionada con una ya elegida
        if kept_pos and np.max(np.abs(C[i, kept_pos])) >= float(f["max_abs_corr"]):
            n_corr += 1
            continue
        kept_pos.append(i)
    kept = order[kept_pos]
    stats.update({"descartadas_por_correlacion": n_corr, "conservadas": int(len(kept)),
                  "conservadas_numericas": int(np.isin(kept, num_idx).sum()),
                  "conservadas_binarias": int(np.isin(kept, bin_idx).sum())})
    return [str(c) for c in cols[kept]], stats


def fit_scaler(Xtr: pd.DataFrame, cfg) -> dict:
    """Calcula la media y la desviacion con el train normal. Las columnas 0/1 no se
    escalan (se deja media 0 y desviacion 1)."""
    A = Xtr.to_numpy(dtype=np.float64)
    is_bin = binary_mask(A)
    # Media y desviacion de cada columna. Si la desviacion es casi 0, se usa 1 para no dividir por 0.
    mu, sd = A.mean(axis=0), A.std(axis=0)
    sd[sd < 1e-12] = 1.0
    # Con media 0 y desviacion 1, las columnas 0/1 quedan iguales
    mu[is_bin], sd[is_bin] = 0.0, 1.0
    if cfg["scaling"]["method"] == "none":
        mu[:], sd[:] = 0.0, 1.0
    return {"columns": list(Xtr.columns), "mean": mu.tolist(), "std": sd.tolist(),
            "binary": is_bin.tolist(), "method": cfg["scaling"]["method"],
            "clip": cfg["scaling"]["clip"]}


# Aplica el escalado calculado con el train normal: (x - media) / desviacion
def apply_scaler(X: pd.DataFrame, sc: dict) -> pd.DataFrame:
    A = (X.to_numpy(dtype=np.float64) - np.array(sc["mean"])) / np.array(sc["std"])
    # Recorta los valores extremos al rango [-clip, clip]
    if sc["clip"] is not None:
        A = np.clip(A, -float(sc["clip"]), float(sc["clip"]))
    return pd.DataFrame(A.astype(np.float32), columns=sc["columns"])


# --------------------------------------------------------------------------- #
# Particiones
# --------------------------------------------------------------------------- #
def stratified_halves(rng, keys: np.ndarray, frac_first: float) -> tuple[np.ndarray, np.ndarray]:
    """Divide los indices en dos partes manteniendo la proporcion de cada grupo
    (normal y cada tipo de ataque). Si un grupo tiene una sola fila, va a la
    segunda parte (la de evaluacion)."""
    first, second = [], []
    # Divide cada grupo por separado, para que ambas partes tengan la misma proporcion
    for g in np.unique(keys):
        idx = rng.permutation(np.flatnonzero(keys == g))
        k = int(np.floor(frac_first * len(idx)))
        first.append(idx[:k])
        second.append(idx[k:])
    return rng.permutation(np.concatenate(first)), rng.permutation(np.concatenate(second))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    sp = cfg["split"]
    seed = int(sp["seed"])
    set_seeds(seed)
    rng = np.random.default_rng(seed)

    # Lee los datos y separa etiquetas, tipos de ataque y variables
    df_tr, df_te = read_inputs(cfg)
    y_tr, y_te = build_target(df_tr, cfg), build_target(df_te, cfg)
    types_te = attack_types(df_te, cfg, y_te)
    X_tr_all, X_te_all = feature_frame(df_tr, cfg), feature_frame(df_te, cfg)
    print(f"\npreprocessor: train {len(df_tr):,} filas (normal {(y_tr == 0).sum():,}, "
          f"ataque {(y_tr == 1).sum():,}) | test {len(df_te):,} filas "
          f"(normal {(y_te == 0).sum():,}, ataque {(y_te == 1).sum():,}) | "
          f"{X_tr_all.shape[1]} columnas numericas")
    # Se detiene si faltan datos normales o una de las clases en el test
    if (y_tr == 0).sum() < 100 or (y_te == 0).sum() == 0 or (y_te == 1).sum() == 0:
        raise SystemExit("Faltan filas normales en el train, o una clase vacia en el test: "
                         "revise dataset.label_col / dataset.normal_value.")

    mode = cfg["features"]["mode"]
    # Avisa si las columnas elegidas por el preprocessor usaron etiquetas
    if mode == "selector" and cfg["features"]["selector"] != "vif":
        print(f"[!] AVISO: el selector '{cfg['features']['selector']}' del profesor usa etiquetas "
              "(y su train incluye ataques). Declarelo como limitacion o use features.mode: "
              "unsupervised.")

    # ---- train / val: solo normal del train del preprocessor -------------------
    # Mezcla las filas normales y separa val del resto, que queda como train
    normal = rng.permutation(np.flatnonzero(y_tr == 0))
    n_val = int(round(float(sp["val_frac_of_normal"]) * len(normal)))
    idx_val, idx_train = normal[:n_val], normal[n_val:]
    Xtrain_raw, Xval_raw = X_tr_all.iloc[idx_train], X_tr_all.iloc[idx_val]

    # ---- caracteristicas y escalado: solo con el train normal ------------------
    # Elige las columnas usando solo el train normal
    if mode == "unsupervised":
        feats, fstats = select_unsupervised(Xtrain_raw.reset_index(drop=True), cfg, seed)
    else:
        feats, fstats = list(X_tr_all.columns), {"conservadas": int(X_tr_all.shape[1])}
    missing = [c for c in feats if c not in X_te_all.columns]
    if missing:
        print(f"[!] AVISO: {len(missing)} columnas del train no estan en el test; se rellenan con 0.")
    # Todos los bloques quedan con las mismas columnas (las que falten en test valen 0)
    Xtrain_raw = Xtrain_raw[feats].reset_index(drop=True)
    Xval_raw = Xval_raw[feats].reset_index(drop=True)
    X_te_sel = X_te_all.reindex(columns=feats, fill_value=0.0)
    # El escalado se calcula con train y se aplica igual a todos los bloques
    scaler = fit_scaler(Xtrain_raw, cfg)
    X_train, X_val = apply_scaler(Xtrain_raw, scaler), apply_scaler(Xval_raw, scaler)
    X_test = apply_scaler(X_te_sel.reset_index(drop=True), scaler)
    print(f"caracteristicas: {len(feats)} ({fstats})")

    # ---- test: (opcional) ajustar la proporcion de ataques; luego dividir en oraculo y final
    keep = np.arange(len(X_test))
    pi = sp.get("test_target_prevalence")
    # Opcional: quita ataques hasta llegar a la proporcion pedida
    if pi:
        att, nor = np.flatnonzero(y_te == 1), np.flatnonzero(y_te == 0)
        # Cantidad de ataques necesaria para llegar a esa proporcion
        want = int(round(float(pi) / (1 - float(pi)) * len(nor)))
        if want < len(att):
            att = rng.choice(att, size=want, replace=False)
        else:
            print(f"[!] No hay ataques suficientes para prevalencia {pi:.0%}: "
                  f"se usan los {len(att):,} disponibles.")
        keep = np.sort(np.concatenate([nor, att]))
    y_k, t_k = y_te[keep], types_te[keep]
    # Divide el test en test_oracle y test_final, manteniendo la proporcion de cada tipo
    o_loc, f_loc = stratified_halves(rng, t_k, float(sp["oracle_frac_of_test"]))
    idx_orc, idx_fin = keep[o_loc], keep[f_loc]

    # ---- escritura ----------------------------------------------------------------
    out = splits_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    ext = io_ext()
    write_table(X_train, out / f"train.{ext}")
    write_table(X_val, out / f"val.{ext}")
    # Solo los bloques de prueba se guardan con etiquetas
    for name, idx in (("test_oracle", idx_orc), ("test_final", idx_fin)):
        d = X_test.iloc[idx].reset_index(drop=True)
        d["y_true"] = y_te[idx]
        d["attack_type"] = types_te[idx]
        write_table(d, out / f"{name}.{ext}")
    # Muestra sin etiquetas que se lleva a la Raspberry
    X_val.head(int(sp["edge_sample_rows"])).to_csv(out / "edge_sample.csv", index=False)
    save_json(scaler, out / "scaler.json")
    save_json({"features": feats, "mode": mode, "stats": fstats}, out / "features.json")

    # Resume un bloque: filas, normales, ataques y cantidad por tipo
    def block(idx):
        yy = y_te[idx]
        return {"rows": int(len(idx)), "normal": int((yy == 0).sum()),
                "attack": int((yy == 1).sum()), "attack_ratio": round(float(yy.mean()), 4),
                "attack_types": {t: int((types_te[idx] == t).sum())
                                 for t in sorted(set(types_te[idx]) - {"normal"})}}

    # Registro de la particion: tamanos, columnas y hashes de cada bloque
    manifest = {
        "dataset": cfg["dataset"]["name"], "input_variant": cfg["input"]["variant"],
        "features_mode": mode, "n_features": len(feats), "feature_names": feats,
        "seed": seed, "format": ext,
        "splits": {
            "train": {"rows": int(len(X_train)), "content": "solo normal, sin etiquetas"},
            "val": {"rows": int(len(X_val)), "content": "solo normal, sin etiquetas"},
            "test_oracle": {**block(idx_orc), "use": "solo escenario oraculo"},
            "test_final": {**block(idx_fin), "use": "solo 5_evaluate.py, una vez"},
        },
        "sha256": {n: sha256_of(out / f"{n}.{ext}")
                   for n in ("train", "val", "test_oracle", "test_final")},
        "note": ("test_final esta aislado: ningun script de optimizacion lo lee. "
                 "5_evaluate.py verifica el hash antes de abrirlo."),
    }
    save_json(manifest, out / "manifest.json")
    save_json(manifest, outputs_dir(cfg) / "logs" / "splits_manifest.json")

    print(f"\nBloques en {out}  (formato {ext})")
    for k, v in manifest["splits"].items():
        extra = f" | ataque {v['attack_ratio']:.1%}" if "attack_ratio" in v else ""
        print(f"  {k:12s}: {v['rows']:>8,} filas{extra}")
    # Avisos si los datos pueden dar resultados enganosos o ruidosos
    ratio = manifest["splits"]["test_final"]["attack_ratio"]
    if ratio > 0.5:
        print(f"\n {ratio:.0%} de test_final es ataque. Con esa prevalencia la Precision "
              "es enganosamente alta y el MCC no representa un despliegue real. Use "
              "split.test_target_prevalence (p. ej. 0.10) o mire MCC_pi05 / MCC_pi10.")
    if manifest["splits"]["test_oracle"]["normal"] < 300:
        print("pocos normales en test_oracle: el FPR estimado sera ruidoso.")
    if len(X_val) < 1000:
        print("+val tiene menos de 1000 filas: umbral y metricas ruidosos. "
              "Considere input.variant: processed.")


if __name__ == "__main__":
    main()
