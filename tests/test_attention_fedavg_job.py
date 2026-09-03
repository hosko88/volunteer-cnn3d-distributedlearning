import numpy as np

from jobs.cnn_3d.attention_job import AttentionCNN3DJob
from jobs.cnn_3d.datasets import SyntheticProvider, LearnableSyntheticProvider


def test_attention_fedavg_job_produces_attention_weighted_update():
    job = AttentionCNN3DJob(volume_shape=(6, 6, 6), n_classes=2)
    params = job.init_params()
    task = job.make_task(epoch=0, index=0, seed=3, epsilon=0.1)

    local_weights, n_samples, metrics = job.compute_local_update(params, task)
    assert local_weights.shape == params.shape
    assert n_samples == task["batch_size"]
    assert metrics["accuracy"] >= 0.0


def test_attention_fedavg_job_compute_gradient_shapes():
    """Exercice le chemin REELLEMENT utilise par le serveur/les volontaires
    (compute_gradient), avec un fournisseur synthetique pour rester rapide
    et sans reseau en test."""
    provider = SyntheticProvider(volume_shape=(6, 6, 6), n_classes=3, seed=1)
    job = AttentionCNN3DJob(volume_shape=(6, 6, 6), n_classes=3,
                             data_provider=provider)
    params = job.init_params()
    task = job.make_task(epoch=0, index=0, seed=42, epsilon=0.1)

    grad, n_samples, metrics = job.compute_gradient(params, task)
    assert grad.shape == params.shape
    assert n_samples == task["batch_size"]
    assert metrics["loss"] >= 0.0


def test_attention_fedavg_job_converges_on_learnable_signal():
    """Verifie une VRAIE capacite de convergence (pas un seul pas sur du
    bruit pur, qui est un test fragile par construction -- cf.
    LearnableSyntheticProvider). Sur ~80 pas de gradient avec un signal
    reellement appris, la perte moyenne doit baisser nettement."""
    provider = LearnableSyntheticProvider(volume_shape=(6, 6, 6), n_classes=3, seed=7)
    job = AttentionCNN3DJob(volume_shape=(6, 6, 6), n_classes=3,
                             n_filters=8, n_hidden=16, data_provider=provider)
    params = job.init_params()
    opt_state = job.init_opt_state()

    losses = []
    for epoch in range(80):
        task = job.make_task(epoch=epoch, index=0, seed=1000 + epoch, epsilon=0.1)
        grad, _, metrics = job.compute_gradient(params, task)
        params, opt_state = job.apply_gradient(params, grad, opt_state, lr=0.001)
        losses.append(metrics["loss"])

    loss_debut = float(np.mean(losses[:10]))
    loss_fin = float(np.mean(losses[-10:]))
    assert loss_fin < loss_debut, (
        f"le modele n'a pas convergé (debut={loss_debut:.4f}, fin={loss_fin:.4f})"
    )
