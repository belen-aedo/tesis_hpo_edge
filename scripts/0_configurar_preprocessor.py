"""
Prepara el preprocessor del profesor para que funcione dentro de este proyecto.
Solo edita sus archivos de configuracion, no sus .py.

Hace dos cambios:
  1) config.yaml: ROOT y RESULTS pasan a apuntar a esta carpeta.
  2) config.json: quita la "/" inicial de la ruta del CSV de Ton-IoT.
     Con esa "/", Windows ignora ROOT y busca el CSV en C:\\Datasets\\...

La primera vez guarda una copia de cada archivo con terminacion .bak_proyecto.

Uso:
    python scripts/0_configurar_preprocessor.py
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

# Carpetas del proyecto y del preprocessor
ROOT = Path(__file__).resolve().parents[1]
PRE = ROOT / "preprocessor"
DEFAULT_REL = "Datasets/Sources/TON-IoT/Train_Test_datasets/Train_Test_Network_dataset/"


def backup(p: Path) -> None:
    """Guarda una copia del archivo, solo si aun no existe."""
    b = p.with_name(p.name + ".bak_proyecto")
    if not b.exists():
        shutil.copy2(p, b)


def yaml_quote(s: str) -> str:
    """Pone el texto entre comillas y convierte las tildes a codigos \\uXXXX.
    Asi la ruta se lee bien aunque Windows abra el archivo con otra codificacion."""
    out = []
    for ch in s:
        if ch in '"\\':
            out.append("\\" + ch)  # comillas y barras: se escapan
        elif ord(ch) < 128:
            out.append(ch)  # caracter normal: queda igual
        elif ord(ch) <= 0xFFFF:
            out.append(f"\\u{ord(ch):04x}")  # tildes y similares
        else:
            out.append(f"\\U{ord(ch):08x}")  # caracteres especiales, como emojis
    return '"' + "".join(out) + '"'


def patch_yaml(text: str, root_posix: str) -> tuple[str, bool]:
    """Cambia ROOT y RESULTS solo en el bloque high_performance.
    Devuelve (texto, si hubo cambio)."""
    lines = text.splitlines(keepends=True)

    # Busca donde empieza el bloque
    start = next((i for i, ln in enumerate(lines) if re.match(r"^  high_performance:\s*$", ln.rstrip("\r\n"))), None)
    if start is None:
        return text, False

    # Busca donde termina: en el siguiente entorno o al final del archivo
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if re.match(r"^  \S", lines[j]):
            end = j
            break

    # Reemplaza las dos lineas y mantiene su salto de linea original
    changed = False
    for k in range(start + 1, end):
        eol = "\r\n" if lines[k].endswith("\r\n") else ("\n" if lines[k].endswith("\n") else "")
        if re.match(r"^    ROOT:", lines[k]):
            lines[k] = f"    ROOT: {yaml_quote(root_posix)}{eol}"
            changed = True
        elif re.match(r"^    RESULTS:", lines[k]):
            lines[k] = f"    RESULTS: {yaml_quote(root_posix + '/RESULTS')}{eol}"
            changed = True
    return "".join(lines), changed


def patch_json(jtext: str) -> tuple[str, str | None, bool]:
    """Quita la "/" inicial del "path" de Ton-IoT.
    Devuelve (texto, ruta final, si hubo cambio). La ruta es None si no la encuentra."""
    # Busca el bloque de Ton-IoT
    m = re.search(r'"Ton-IoT"\s*:\s*\{', jtext)
    if not m:
        return jtext, None, False

    # El bloque termina donde empieza el siguiente dataset
    start = m.end()
    nxt = re.search(r'\n {4}"[^"]+"\s*:\s*\{', jtext[start:])
    end = start + nxt.start() if nxt else len(jtext)
    block = jtext[start:end]

    # Busca la ruta dentro del bloque
    pm = re.search(r'("path"\s*:\s*")([^"]*)(")', block)
    if not pm:
        return jtext, None, False

    # Si no tenia "/" inicial, no hay nada que cambiar
    old = pm.group(2)
    new = old.lstrip("/\\")
    if new == old:
        return jtext, new, False

    # Reemplaza solo la ruta y deja el resto del texto igual
    block2 = block[:pm.start(2)] + new + block[pm.end(2):]
    return jtext[:start] + block2 + jtext[end:], new, True


def main() -> int:
    # 1. Revisa que el preprocessor este copiado en el proyecto
    yml, js, v2 = PRE / "config.yaml", PRE / "config.json", PRE / "preprocessor_v2"
    faltan = [str(p.relative_to(ROOT)) for p in (yml, js, v2) if not p.exists()]
    if faltan:
        print("Falta copiar el preprocessor del profesor dentro de la carpeta 'preprocessor/'.\n"
              "Faltan: " + ", ".join(faltan) + "\n"
              "Descomprima preprocessor.zip y copie el CONTENIDO de su carpeta 'preprocessor' "
              "(preprocessor_v2/, config.yaml, config.json, ...) dentro de 'preprocessor/'.")
        return 1

    # 2. Corrige config.yaml
    root_posix = ROOT.as_posix()
    with open(yml, "r", encoding="utf-8", newline="") as fh:  # newline="" mantiene los saltos de linea
        text = fh.read()
    new, changed = patch_yaml(text, root_posix)
    if not changed:
        print("[!] No encontre ROOT/RESULTS en el entorno high_performance de config.yaml; "
              "editelos a mano:  ROOT: \"" + root_posix + "\"")
        return 1
    backup(yml)
    with open(yml, "w", encoding="utf-8", newline="") as fh:
        fh.write(new)
    print(f"config.yaml  -> ROOT = {root_posix}")

    # 3. Corrige config.json
    with open(js, "r", encoding="utf-8", newline="") as fh:
        jtext = fh.read()
    new_text, rel, changed = patch_json(jtext)
    if rel is None:
        print('[!] No encontre el "path" del dataset Ton-IoT en config.json; debe quedar SIN "/" '
              'inicial, p. ej. "Datasets/Sources/TON-IoT/Train_Test_datasets/'
              'Train_Test_Network_dataset/preprocessed.csv".')
        return 1
    if changed:
        backup(js)
        with open(js, "w", encoding="utf-8", newline="") as fh:
            fh.write(new_text)
        print(f"config.json  -> path de Ton-IoT sin '/' inicial: {rel}")
    else:
        print(f"config.json  -> ya estaba corregido: {rel}")

    # 4. Avisa si el dataset ya esta en su carpeta (si falta, no es un error)
    folder = (ROOT / rel).parent
    raw = next((p for p in folder.glob("*.csv") if p.name.lower() == "train_test_network.csv"), None) \
        if folder.exists() else None
    lab = folder / "Train_Test_Labels.csv"
    if raw is not None or lab.exists():
        f = raw or lab
        print(f"\nDataset encontrado: {f}  ({f.stat().st_size / 1e6:.0f} MB)")
    else:
        print(f"\n[!] Falta el dataset. Copie Train_Test_Network.csv en:\n    {folder}\n"
              "(lo descarga de la pagina oficial de TON_IoT de UNSW, carpeta "
              "Train_Test_datasets/Train_Test_Network_dataset).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
