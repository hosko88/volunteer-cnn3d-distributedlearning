import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEFAULT
from jobs.cnn_3d.attention_job import AttentionCNN3DJob
from server.app import create_app


class AutoStartServerTest(unittest.TestCase):
    def test_auto_start_sets_training_started(self):
        cfg = DEFAULT
        job = AttentionCNN3DJob()
        app = create_app(job, cfg, auto_start=True)

        with app.test_client() as client:
            response = client.get("/training_status")
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.get_json()["started"])


if __name__ == "__main__":
    unittest.main()
