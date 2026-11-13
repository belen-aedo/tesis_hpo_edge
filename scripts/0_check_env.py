"""
0_check_env.py - revisa que esten instaladas las librerias y muestra sus versiones.

    python scripts/0_check_env.py            # en el PC
    python scripts/0_check_env.py --rol pi   # en la Raspberry (ahi no se piden optuna, tensorflow ni pyarrow)

Importante: scikit-learn y numpy tienen que tener la misma version en el PC y en
la Raspberry. Los modelos se guardan con pickle y si las versiones no coinciden,
pueden no cargar.
"""

from __future__ import annotations

import argparse
import importlib
import platform
import sys

# Librerias necesarias en el PC y en la Raspberry
REQ_PC = ["numpy", "pandas", "scipy", "sklearn", "yaml", "psutil", "optuna", "pyarrow"]
REQ_PRE = ["lightgbm", "statsmodels"]         # solo si usas features.mode: selector (paso 04)
REQ_PI = ["numpy", "pandas", "scipy", "sklearn", "yaml", "psutil"]
OPCIONAL = ["tensorflow"]                     # solo para entrenar el autoencoder con Keras en el PC


# Devuelve la version de una libreria, o None si no esta instalada
def ver(mod: str):
    try:
        m = importlib.import_module(mod)
        return getattr(m, "__version__", "?")
    except Exception:
        return None


# Muestra que librerias estan y avisa si falta alguna. Devuelve 1 si falta alguna obligatoria.
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rol", choices=["pc", "pi"], default="pc")
    args = ap.parse_args()
    print(f"Python {platform.python_version()} | {platform.platform()}")
    req = REQ_PI if args.rol == "pi" else REQ_PC
    faltan = []
    # Revisa las librerias obligatorias
    for m in req:
        v = ver(m)
        print(f"  {'OK ' if v else 'FALTA'} {m:12s} {v or ''}")
        if not v:
            faltan.append(m)
    # En el PC, muestra tambien las opcionales. Estas no provocan error.
    if args.rol == "pc":
        for m in REQ_PRE + OPCIONAL:
            v = ver(m)
            print(f"  {'ok ' if v else '-- '} {m:12s} {v or '(opcional)'}")
    if faltan:
        print("\nInstale:  pip install -r " + ("requirements-pi.txt" if args.rol == "pi"
                                               else "requirements.txt"))
        return 1
    print("\nTodo listo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
