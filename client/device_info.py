"""
Detection des caracteristiques de l'appareil volontaire
=======================================================
Sert a (1) annoncer le type d'appareil au serveur et (2) estimer sa PUISSANCE,
qui determine combien de sous-taches lui sont confiees par requete.
"""

import os
import platform
import multiprocessing
import time
import math


def is_android():
    return "ANDROID_ROOT" in os.environ or "ANDROID_DATA" in os.environ


def get_ram_gb():
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    except (ValueError, OSError, AttributeError):
        return None


def get_device_info(label=None):
    cpu = multiprocessing.cpu_count()
    osname = "android" if is_android() else platform.system().lower()
    return {
        "device": label or osname,
        "os": osname,
        "cpu": cpu,
        "ram_gb": get_ram_gb(),
        "python": platform.python_version(),
        "machine": platform.machine(),
    }

def benchmark_2s(duration=2.0):
    """
    Benchmark CPU court exécuté au lancement du volontaire.
    Il permet d'estimer automatiquement la puissance réelle de l'appareil.
    """
    start = time.time()
    ops = 0
    x = 0.0

    while time.time() - start < duration:
        for i in range(1000):
            x += math.sin(i) * math.cos(i)
        ops += 1000

    elapsed = time.time() - start
    return int(ops / elapsed)

def estimate_power(info):
    """
    Puissance initiale provisoire.
    On évite de plafonner tout le monde à 4.
    """
    score = info.get("benchmark_score", 0)

    if score <= 0:
        return 1

    # Exemple : 2 000 000 -> puissance 2
    # 4 000 000 -> puissance 4
    # 8 000 000 -> puissance 8
    return max(1, min(12, round(score / 1_000_000)))
