"""
objectives.py - funciones objetivo no supervisadas.

Ninguna funcion de este archivo recibe etiquetas. Todas se calculan sobre la
mitad `objetivo` del bloque `val` (solo trafico normal).

    silhouette      maximizar   silhouette sobre los puntajes (1 dimension), con
                                los 2 grupos que separa el umbral tau_q
    davies_bouldin  minimizar   lo mismo, pero con Davies-Bouldin
    reconstruction  minimizar   error medio de reconstruccion (la opcion natural
                                para el autoencoder)
    stability       maximizar   Spearman promedio entre n copias de los datos con
                                ruido gaussiano
    consistency     maximizar   Spearman promedio entre modelos reentrenados con
                                distintas submuestras del train

Los escenarios `oracle` y `oracle_q` (cota superior) maximizan el MCC con las
etiquetas de test_oracle. Se calculan en 3_optimize.py, no aqui.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import davies_bouldin_score, silhouette_score

# Indica si cada criterio se maximiza o se minimiza
DIRECTION = {
    "silhouette": "maximize",
    "davies_bouldin": "minimize",
    "reconstruction": "minimize",
    "stability": "maximize",
    "consistency": "maximize",
    "oracle": "maximize",       # MCC con etiquetas de test_oracle, q fijo
    "oracle_q": "maximize",     # igual, pero q tambien se busca
}
# Criterios que no usan etiquetas
UNSUPERVISED = ("silhouette", "davies_bouldin", "reconstruction", "stability", "consistency")
# Valor malo que recibe un trial cuando su criterio no se puede calcular
PENALTY = {"maximize": -1.0, "minimize": 1e6}


# Devuelve la penalizacion segun si el criterio se maximiza o se minimiza
def penalty_for(metric: str) -> float:
    return PENALTY[DIRECTION[metric]]


# Prepara lo que usan Silhouette y Davies-Bouldin:
#   z    = puntajes estandarizados (media 0, desviacion 1)
#   part = grupo de cada fila: 1 si supera el umbral, 0 si no
# Ojo: aqui solo hay trafico normal, asi que los grupos separan los puntajes mas
# altos del resto, no el trafico normal de los ataques.
def _score_space(det, X):
    s = np.asarray(det.scores(X), dtype=np.float64).ravel()
    sd = s.std()
    # Si todos los puntajes son iguales (sd = 0), z queda en ceros
    z = (s - s.mean()) / sd if sd > 0 else s * 0.0
    return z.reshape(-1, 1), (s > det.threshold_).astype(int)


# Silhouette: que tan separados estan los 2 grupos de puntajes (mas alto = mejor)
def objective_silhouette(det, X, cfg) -> float:
    z, part = _score_space(det, X)
    # Si todas las filas cayeron en un solo grupo, no se puede calcular
    if len(np.unique(part)) < 2:
        return penalty_for("silhouette")
    # Usa una muestra de filas para que el calculo no sea tan lento
    n = min(len(z), int(cfg["optimization"]["silhouette_sample"]))
    try:
        return float(silhouette_score(z, part, sample_size=n, random_state=0))
    except ValueError:
        return penalty_for("silhouette")


# Davies-Bouldin: compara la dispersion de cada grupo con la distancia entre
# ellos (mas bajo = mejor)
def objective_davies_bouldin(det, X, cfg) -> float:
    z, part = _score_space(det, X)
    # Si todas las filas cayeron en un solo grupo, no se puede calcular
    if len(np.unique(part)) < 2:
        return penalty_for("davies_bouldin")
    return float(davies_bouldin_score(z, part))


# Promedio de los puntajes en el bloque objetivo (mas bajo = mejor).
# En el autoencoder, ese puntaje es el error de reconstruccion.
def objective_reconstruction(det, X, cfg) -> float:
    return float(np.mean(det.scores(X)))


# Compara cada par de vectores de puntajes con Spearman y promedia.
# Spearman mira si ambos ordenan las filas igual, no si los valores son iguales.
# Devuelve None si no pudo comparar ningun par.
def _mean_pairwise_spearman(vectors) -> float | None:
    corrs = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            # Spearman no se puede calcular con puntajes constantes: se salta ese par
            if np.std(vectors[i]) == 0 or np.std(vectors[j]) == 0:
                continue
            rho = spearmanr(vectors[i], vectors[j]).statistic
            if np.isfinite(rho):
                corrs.append(rho)
    return float(np.mean(corrs)) if corrs else None


def objective_stability(det, X, cfg, seed: int = 42) -> float:
    """Estabilidad del ranking: se agrega ruido gaussiano al bloque objetivo varias
    veces (sigma en unidades estandarizadas) y se promedia el Spearman entre los
    puntajes de cada copia."""
    o = cfg["objectives"]
    rng = np.random.default_rng(seed)
    # Una lista de puntajes por cada copia de los datos con ruido
    reps = [np.asarray(det.scores(X + rng.normal(0.0, o["stability_noise_sigma"], X.shape)
                                  .astype(X.dtype)), dtype=np.float64).ravel()
            for _ in range(int(o["stability_replicas"]))]
    v = _mean_pairwise_spearman(reps)
    return penalty_for("stability") if v is None else v


def objective_consistency(det, X, cfg, ctx, seed: int = 42) -> float:
    """Consistencia al cambiar los datos: se reentrena la misma configuracion con
    `consistency_boot` submuestras distintas del train y se promedia el Spearman
    entre los puntajes que cada modelo le da al bloque objetivo."""
    if ctx is None:
        raise ValueError("consistency requiere ctx={'Detector','params','X_train'}")
    o = cfg["optimization"]
    rng = np.random.default_rng(seed)
    Xtr = ctx["X_train"]
    # Cantidad de filas de cada submuestra
    k = max(int(o["consistency_frac"] * len(Xtr)), 10)
    vecs = []
    # Reentrena el modelo con una submuestra distinta y guarda sus puntajes
    for _ in range(int(o["consistency_boot"])):
        p = dict(ctx["params"])
        # Tambien cambia la semilla del modelo, si tiene una
        if "random_state" in p:
            p["random_state"] = int(rng.integers(0, 2**31 - 1))
        d = ctx["Detector"](**p).fit(Xtr[rng.choice(len(Xtr), size=k, replace=False)])
        vecs.append(np.asarray(d.scores(X), dtype=np.float64).ravel())
    v = _mean_pairwise_spearman(vecs)
    return penalty_for("consistency") if v is None else v


# Llama a la funcion que corresponde al criterio pedido
def compute_objective(metric: str, det, X, cfg: Dict[str, Any], ctx=None) -> float:
    if metric == "silhouette":
        return objective_silhouette(det, X, cfg)
    if metric == "davies_bouldin":
        return objective_davies_bouldin(det, X, cfg)
    if metric == "reconstruction":
        return objective_reconstruction(det, X, cfg)
    if metric == "stability":
        return objective_stability(det, X, cfg)
    if metric == "consistency":
        return objective_consistency(det, X, cfg, ctx)
    raise ValueError(f"Metrica no supervisada desconocida: {metric}. Opciones: {UNSUPERVISED}")
