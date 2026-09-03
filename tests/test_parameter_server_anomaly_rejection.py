import numpy as np

from config import DEFAULT
from framework.parameter_server import ParameterServer
from framework.compression import encode_vector
from jobs.cnn_3d.fedavg_job import FedAvgCNN3DJob


def test_parameter_server_rejects_outlier_gradients_as_anomalies():
    cfg = DEFAULT
    job = FedAvgCNN3DJob(volume_shape=(4, 4, 4), n_classes=2)
    ps = ParameterServer(job, cfg.transport)

    theta, _ = ps.get_theta()
    bad_grad = np.ones(job.n_params(), dtype=np.float32) * 1e9
    payload, _, _ = encode_vector(bad_grad, dtype=cfg.transport.dtype, topk=1.0)

    ok, staleness = ps.assimilate(payload, 0, 1, cfg.model.lr, cfg.train.staleness_max)

    assert ok is False
    assert staleness == 0
    stats = ps.bandwidth_stats()
    assert stats["gradients_anomalies_rejetes"] == 1
