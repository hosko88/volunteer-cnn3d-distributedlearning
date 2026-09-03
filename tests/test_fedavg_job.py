import numpy as np

from jobs.cnn_3d.fedavg_job import FedAvgCNN3DJob


def test_fedavg_job_returns_local_weights_and_aggregates_them():
    job = FedAvgCNN3DJob(volume_shape=(6, 6, 6), n_classes=2)
    params = job.init_params()
    task = job.make_task(epoch=0, index=0, seed=7, epsilon=0.1)

    local_weights, n_samples, metrics = job.compute_local_update(params, task)
    assert local_weights.shape == params.shape
    assert n_samples == task["batch_size"]
    assert metrics["accuracy"] >= 0.0

    aggregated, _ = job.aggregate_local_update(
        params,
        local_weights,
        None,
        0.01,
        n_samples,
        client_info={"benchmark_score": 1.4},
    )

    assert aggregated.shape == params.shape
    assert not np.allclose(aggregated, params)
