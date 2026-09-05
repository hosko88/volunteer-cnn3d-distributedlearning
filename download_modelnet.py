"""
Telechargement robuste de ModelNet40 (avec retry + reprise) via requests,
en remplacement du telechargement urllib fragile du projet.
A executer depuis la racine du projet (dossier volunteer_cnn3d_distributed),
dans le terminal integre de VS Code, avec le venv active.
"""
import os
import time
import requests
from pathlib import Path

DATA_ROOT = Path("data")
DATA_ROOT.mkdir(parents=True, exist_ok=True)
DEST = DATA_ROOT / "modelnet40_ply_hdf5_2048.zip"

# Miroirs a essayer dans l'ordre (le premier public/fiable trouve)
URLS = [
    "https://huggingface.co/datasets/Msun/modelnet40/resolve/main/modelnet40_ply_hdf5_2048.zip",
    "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip",
]

def download(url, dest, max_retries=5):
    headers = {}
    mode = "wb"
    resume_pos = 0
    if dest.exists():
        resume_pos = dest.stat().st_size
        headers["Range"] = f"bytes={resume_pos}-"
        mode = "ab"

    for attempt in range(1, max_retries + 1):
        try:
            print(f"[tentative {attempt}/{max_retries}] {url}")
            with requests.get(url, headers=headers, stream=True, timeout=30) as r:
                if r.status_code == 416:
                    print("Deja complet.")
                    return True
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0)) + resume_pos
                downloaded = resume_pos
                with open(dest, mode) as f:
                    for chunk in r.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total:
                                pct = downloaded * 100 / total
                                print(f"\r  {pct:5.1f}% ({downloaded/1e6:.1f} Mo / {total/1e6:.1f} Mo)", end="", flush=True)
                print()
                return True
        except Exception as e:
            print(f"  echec : {e}")
            resume_pos = dest.stat().st_size if dest.exists() else 0
            headers["Range"] = f"bytes={resume_pos}-"
            mode = "ab"
            time.sleep(3)
    return False

if __name__ == "__main__":
    if DEST.exists() and DEST.stat().st_size > 400_000_000:
        print(f"Fichier deja present et complet : {DEST} ({DEST.stat().st_size/1e6:.1f} Mo)")
    else:
        ok = False
        for url in URLS:
            if download(url, DEST):
                ok = True
                break
        if not ok:
            raise SystemExit("Echec du telechargement sur tous les miroirs.")

    print(f"OK TELECHARGEMENT TERMINE : {DEST} ({DEST.stat().st_size/1e6:.1f} Mo)")