"""
detectors.py - los tres detectores: Isolation Forest, LOF y Autoencoder.

Cada detector entrega un puntaje s(x): mientras mas alto, mas anomalo. Un flujo
se marca como ataque si s(x) > tau, donde tau es el percentil q de los puntajes
de la mitad de calibracion de `val` (solo trafico normal).

q se fija de antemano (threshold.q en la config) y no es un hiperparametro. Asi
el optimizador no puede "hacer trampa" moviendo el umbral: cuando `contamination`
estaba libre, Silhouette y Davies-Bouldin terminaban en la solucion inutil de
marcar casi nada como anomalia.

Autoencoder en el borde: se entrena con Keras (si TensorFlow esta instalado) o con
scikit-learn, pero la prediccion siempre se hace con NumPy usando los pesos
guardados. Asi la Raspberry no necesita TensorFlow y se evita el costo fijo de
`model.predict` (~50 ms por muestra).
"""

from __future__ import annotations

import os
import pickle
import warnings

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Menos mensajes de TensorFlow en la consola (no cambia los resultados de forma relevante)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")


# Lo que comparten los tres detectores: calcular el umbral y marcar ataques
class BaseDetector:
    name = ""
    q = 0.95
    threshold_ = None
    calib_scores_ = None

    def scores(self, X: np.ndarray) -> np.ndarray:  # mas alto = mas anomalo
        raise NotImplementedError

    def set_threshold(self, X_calib: np.ndarray, q: float | None = None):
        if q is not None:
            self.q = float(q)
        # Guarda los puntajes de calibracion, para poder calcular umbrales con otros q
        self.calib_scores_ = np.sort(np.asarray(self.scores(X_calib), dtype=np.float64))
        # El umbral es el percentil q de los puntajes de trafico normal
        self.threshold_ = float(np.quantile(self.calib_scores_, self.q))
        return self

    # Calcula el umbral para otro q, sin cambiar el umbral guardado
    def threshold_for(self, q: float) -> float:
        return float(np.quantile(self.calib_scores_, q))

    # Marca cada fila: 1 = ataque (puntaje sobre el umbral), 0 = normal
    def predict_outliers(self, X: np.ndarray, q: float | None = None) -> np.ndarray:
        thr = self.threshold_ if q is None else self.threshold_for(q)
        return (self.scores(X) > thr).astype(int)

    def predict_native(self, X: np.ndarray) -> np.ndarray:
        """Decision que toma la libreria con su configuracion por defecto. Solo se usa
        como referencia en la linea base nativa."""
        raise NotImplementedError


# Isolation Forest: aisla cada fila con cortes al azar. Lo facil de aislar es anomalo.
class IForestDetector(BaseDetector):
    name = "iforest"

    def __init__(self, **params):
        params = dict(params)
        params.pop("contamination", None)        # no se usa: el umbral es el percentil q
        # Separa la semilla del resto de los hiperparametros
        self.random_state = params.pop("random_state", 42)
        self.params = params

    def fit(self, X: np.ndarray) -> "IForestDetector":
        p = dict(self.params)
        # max_samples no puede ser mayor que la cantidad de filas de train
        ms = p.get("max_samples", "auto")
        if isinstance(ms, (int, np.integer)):
            p["max_samples"] = int(min(ms, len(X)))
        self.model = IsolationForest(random_state=self.random_state, n_jobs=1, **p)
        self.model.fit(X)
        return self

    def scores(self, X):
        # Se cambia el signo: scikit-learn da valores mas bajos a lo mas anomalo,
        # y aqui se usa mas alto = mas anomalo.
        return -self.model.score_samples(X)

    def predict_native(self, X):
        # scikit-learn devuelve -1 para anomalia. Se convierte a 1 = ataque.
        return (self.model.predict(X) == -1).astype(int)


class LOFDetector(BaseDetector):
    """LOF en modo novelty: aprende con trafico normal y despues evalua datos nuevos.
    Guarda todo el train, asi que la RAM y la latencia crecen con su tamano; por
    eso common.train_for le pasa menos filas."""
    name = "lof"

    def __init__(self, **params):
        params = dict(params)
        params.pop("contamination", None)
        params.pop("random_state", None)
        self.params = params

    def fit(self, X):
        # novelty=True permite puntuar datos nuevos despues de entrenar
        self.model = LocalOutlierFactor(novelty=True, n_jobs=1, **self.params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)      # aviso por filas duplicadas en el train
            self.model.fit(X)
        return self

    def scores(self, X):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            # Se cambia el signo para usar mas alto = mas anomalo
            return -self.model.score_samples(X)

    def predict_native(self, X):
        # scikit-learn devuelve -1 para anomalia. Se convierte a 1 = ataque.
        return (self.model.predict(X) == -1).astype(int)


# Revisa si TensorFlow esta instalado
def tf_available() -> bool:
    try:
        import tensorflow  # noqa: F401
        return True
    except Exception:
        return False


class AutoencoderDetector(BaseDetector):
    """Autoencoder denso entrenado solo con trafico normal. El puntaje es el error
    cuadratico medio de cada fila."""
    name = "autoencoder"

    # Valores por defecto: son los hiperparametros de la linea base del autoencoder
    def __init__(self, n_layers=2, units=64, latent_dim=8, learning_rate=1e-3,
                 batch_size=256, dropout=0.0, epochs=30, random_state=42,
                 backend="auto", **_ignored):
        self.cfg = dict(n_layers=n_layers, units=units, latent_dim=latent_dim,
                        learning_rate=learning_rate, batch_size=batch_size,
                        dropout=dropout, epochs=epochs, random_state=random_state)
        self.backend = backend
        self.layers_: list = []
        self.native_threshold_ = None

    # Tamano de las capas del encoder: units, units/2, units/4...
    # Cada capa queda siempre mas grande que la capa central.
    def _sizes(self):
        c = self.cfg
        return [max(c["latent_dim"] + 1, int(c["units"] / (2 ** i))) for i in range(c["n_layers"])]

    # Entrena el autoencoder con Keras y guarda solo sus pesos
    def _fit_keras(self, X):
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        c = self.cfg
        tf.random.set_seed(c["random_state"])
        sizes = self._sizes()
        inp = keras.Input(shape=(X.shape[1],))
        h = inp
        # Encoder: comprime la entrada
        for s in sizes:
            h = layers.Dense(s, activation="relu")(h)
            if c["dropout"] > 0:
                h = layers.Dropout(c["dropout"])(h)
        # Capa central: la version mas comprimida de los datos
        h = layers.Dense(c["latent_dim"], activation="relu")(h)
        # Decoder: reconstruye la entrada, con las capas en orden inverso
        for s in reversed(sizes):
            h = layers.Dense(s, activation="relu")(h)
        out = layers.Dense(X.shape[1], activation="linear")(h)
        model = keras.Model(inp, out)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=c["learning_rate"]), loss="mse")
        # La entrada y la salida esperada son las mismas: aprende a copiar trafico normal
        model.fit(X, X, epochs=c["epochs"], batch_size=c["batch_size"], shuffle=True, verbose=0)
        w = model.get_weights()                       # [W1, b1, W2, b2, ...]
        self.layers_ = [(w[i].astype(np.float32), w[i + 1].astype(np.float32))
                        for i in range(0, len(w), 2)]
        try:                                           # limpia la sesion para no acumular grafos entre trials
            keras.backend.clear_session()
        except Exception:
            pass

    # Entrena la misma arquitectura con scikit-learn y guarda solo sus pesos
    def _fit_sklearn(self, X):
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.neural_network import MLPRegressor
        c = self.cfg
        sizes = self._sizes()
        # Red con forma de reloj de arena: encoder + capa central + decoder
        mlp = MLPRegressor(hidden_layer_sizes=tuple(sizes + [c["latent_dim"]] + sizes[::-1]),
                           activation="relu", solver="adam",
                           learning_rate_init=c["learning_rate"],
                           batch_size=int(min(c["batch_size"], len(X))),
                           max_iter=int(c["epochs"]), tol=1e-6, shuffle=True,
                           random_state=c["random_state"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            mlp.fit(X, X)       # ojo: con scikit-learn no se aplica dropout
        self.layers_ = [(W.astype(np.float32), b.astype(np.float32))
                        for W, b in zip(mlp.coefs_, mlp.intercepts_)]

    def fit(self, X):
        # Elige la libreria de entrenamiento segun `backend`
        use_keras = self.backend == "keras" or (self.backend == "auto" and tf_available())
        (self._fit_keras if use_keras else self._fit_sklearn)(X)
        # Umbral de la linea base nativa: percentil 95 de los errores en train
        self.native_threshold_ = float(np.percentile(self.scores(X), 95.0))
        return self

    # Pasa los datos por la red usando solo NumPy (asi la Raspberry no necesita TensorFlow)
    def _forward(self, X: np.ndarray) -> np.ndarray:
        h = X.astype(np.float32, copy=False)
        for i, (W, b) in enumerate(self.layers_):
            # Capa: multiplica por los pesos y suma el sesgo
            h = h @ W + b
            if i < len(self.layers_) - 1:
                # Activacion ReLU: los negativos pasan a 0 (no se aplica en la capa de salida)
                np.maximum(h, 0.0, out=h)
        return h

    # Puntaje = error de reconstruccion: mientras peor copia una fila, mas anomala es
    def scores(self, X):
        X = np.asarray(X, dtype=np.float32)
        out = np.empty(len(X), dtype=np.float64)
        # Procesa por partes de 8192 filas para no usar demasiada memoria
        for s in range(0, len(X), 8192):
            xb = X[s:s + 8192]
            out[s:s + 8192] = np.mean((xb - self._forward(xb)) ** 2, axis=1)
        return out

    def predict_native(self, X):
        return (self.scores(X) > self.native_threshold_).astype(int)


# Permite obtener la clase de un detector a partir de su nombre
DETECTORS = {"iforest": IForestDetector, "lof": LOFDetector, "autoencoder": AutoencoderDetector}


# Tamano del modelo guardado, en KB
def model_size_kb(det) -> float:
    try:
        return len(pickle.dumps(det, protocol=pickle.HIGHEST_PROTOCOL)) / 1024.0
    except Exception:
        return float("nan")
