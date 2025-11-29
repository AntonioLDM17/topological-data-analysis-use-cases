import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import wfdb
import wfdb.processing as wfproc
from ripser import ripser
from persim import plot_diagrams

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. Descargar y cargar ECG real (con remuestreo)
# ============================================================

def load_ecg_segment(record_name="16265", pn_dir="nsrdb",
                     fs_target=128, duration_seconds=30, channel=0):
    """
    Carga un segmento de ECG y, si hace falta, remuestrea
    para que tenga frecuencia fs_target.
    """
    record = wfdb.rdrecord(record_name, pn_dir=pn_dir)
    sig = record.p_signal
    fs_orig = int(record.fs)

    # Nos quedamos solo con el canal que queremos (1D)
    x = sig[:, channel]

    # 1) Remuestreo si hace falta
    if fs_orig != fs_target:
        # remuestrea la señal al nuevo fs_target
        x_resampled, _ = wfproc.resample_sig(
            x, fs_orig, fs_target
        )
        fs = fs_target
        x = x_resampled
    else:
        fs = fs_orig

    n_samples_total = len(x)
    n_samples_segment = int(duration_seconds * fs)

    if n_samples_segment > n_samples_total:
        raise ValueError("La duración pedida es mayor que la del registro remuestreado.")

    ecg = x[:n_samples_segment]
    t = np.linspace(0, duration_seconds, n_samples_segment, endpoint=False)

    return t, ecg, fs


# ============================================================
# Sliding window embedding
# ============================================================

def sliding_window(x, dim, tau):
    N = len(x)
    M = N - (dim - 1) * tau
    if M <= 0:
        raise ValueError("Serie demasiado corta para (dim, tau).")

    windows = np.stack(
        [x[i:i + M] for i in range(0, dim * tau, tau)],
        axis=1
    )
    return windows


# ============================================================
# Funciones de visualización (guardan imágenes)
# ============================================================

def plot_time_series(t, ecg, title, save_path):
    plt.figure(figsize=(12, 4))
    plt.plot(t, ecg)
    plt.title(title)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud (mV)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_embedding_2d(X, title, save_path):
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)

    plt.figure(figsize=(5, 5))
    plt.scatter(X2[:, 0], X2[:, 1], s=5)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_persistence_diagrams(dgms, title_prefix, save_path):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plot_diagrams(dgms[0], show=False)
    plt.title(f"{title_prefix} - H0")

    plt.subplot(1, 2, 2)
    if len(dgms) > 1:
        plot_diagrams(dgms[1], show=False)
        plt.title(f"{title_prefix} - H1")
    else:
        plt.text(0.5, 0.5, "Sin H1", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ============================================================
# Homología persistente
# ============================================================

def compute_persistence(X, maxdim=1):
    result = ripser(X, maxdim=maxdim)
    return result["dgms"]


# ============================================================
# PIPELINE completo para un registro individual
# ============================================================

def process_ecg_record(record_name, pn_dir, tag, duration_seconds=30, dim=30, tau=2):
    """
    Procesa un registro real (normal o arrítmico) y guarda todas las imágenes.
    """
    print(f"\n=== Procesando {tag} ({record_name}, {pn_dir}) ===")

    # 1) Cargar señal real
    t, ecg, fs = load_ecg_segment(record_name, pn_dir, duration_seconds)
    print(f"Cargado registro {record_name}. Frecuencia: {fs} Hz")

    # 2) Guardar señal
    ts_path = os.path.join(OUTPUT_DIR, f"{tag}_timeseries.png")
    plot_time_series(t, ecg, f"{tag} - Serie temporal", ts_path)
    print(f"[OK] Serie temporal guardada en {ts_path}")

    # 3) Embedding
    X = sliding_window(ecg, dim, tau)
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    # 4) Guardar embedding PCA 2D
    emb_path = os.path.join(OUTPUT_DIR, f"{tag}_embedding.png")
    plot_embedding_2d(X_scaled, f"{tag} - Embedding PCA", emb_path)
    print(f"[OK] Embedding guardado en {emb_path}")

    # 5) Homología persistente
    dgms = compute_persistence(X_scaled, maxdim=1)

    # 6) Guardar diagramas de persistencia
    dgm_path = os.path.join(OUTPUT_DIR, f"{tag}_persistence.png")
    plot_persistence_diagrams(dgms, f"{tag}", dgm_path)
    print(f"[OK] Persistencia guardada en {dgm_path}")

    # 7) Calcular persistencia máxima en H1
    H1 = dgms[1] if len(dgms) > 1 else np.empty((0, 2))
    if len(H1) > 0:
        persistence = H1[:, 1] - H1[:, 0]
        max_pers = persistence.max()
    else:
        max_pers = 0.0

    print(f"[RESULTADO] Persistencia máxima H1 ({tag}): {max_pers:.4f}")
    return max_pers


# ============================================================
# EJECUCIÓN: NORMAL vs ARRÍTMICO
# ============================================================

if __name__ == "__main__":
    print("Iniciando comparación Normal vs Arrítmico...\n")

    # NORMAL (NSRDB, ritmo sinusal normal)
    pers_normal = process_ecg_record(
        record_name="16265",
        pn_dir="nsrdb",
        tag="normal",
        duration_seconds=30,
        dim=30,
        tau=2
    )

    # ARRÍTMICO (MIT-BIH Arrhythmia Database)
    pers_arr = process_ecg_record(
        record_name="100",
        pn_dir="mitdb",
        tag="arritmico",
        duration_seconds=30,
        dim=30,
        tau=2
    )

    print("\n====================")
    print("   COMPARACIÓN FINAL")
    print("====================")
    print(f"Persistencia H1 (normal)    = {pers_normal:.4f}")
    print(f"Persistencia H1 (arrítmico) = {pers_arr:.4f}")

    if pers_normal > pers_arr:
        print("\nInterpretación: El corazón NORMAL muestra un ciclo topológico más fuerte (H1 persistente).")
        print("Esto es consistente con una periodicidad más estable del latido.")
    else:
        print("\nInterpretación: El registro ARRÍTMICO tiene un ciclo más fuerte, lo cual sería raro.")
        print("Probablemente la señal arrítmica rompe la estructura cíclica, como se esperaba.")
