import pytest

pytest.importorskip("apscheduler")
from hyperion3.config.base_config import HyperionV2Config
from scripts.data_scheduler import start_scheduler


def test_start_scheduler_creates_job():
    cfg = HyperionV2Config()
    sched = start_scheduler(cfg)
    try:
        assert sched.running
        assert len(sched.get_jobs()) == 1
    finally:
        sched.shutdown()
