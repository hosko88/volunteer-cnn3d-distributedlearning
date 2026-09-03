import numpy as np

from jobs.cnn_3d.job import CNN3DJob
from jobs.cnn_3d.datasets import SyntheticProvider


def test_cnn3d_job_returns_gradient_and_metrics():
    # fournisseur synthetique injecte : pas de reseau / telechargement en test
    provider = SyntheticProvider(volume_shape=(6, 6, 6), n_classes=2, seed=0)
    job = CNN3DJob(volume_shape=(6, 6, 6), n_classes=2, data_provider=provider)
    params = job.init_params()
    task = job.make_task(epoch=0, index=0, seed=7, epsilon=0.1)
    grad, n_samples, metrics = job.compute_gradient(params, task)

    assert grad.shape == params.shape
    assert n_samples == task["batch_size"]
    assert metrics["accuracy"] >= 0.0
    assert metrics["loss"] >= 0.0
