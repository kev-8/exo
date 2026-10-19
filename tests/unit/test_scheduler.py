"""Unit tests for ExoScheduler job registration — no jobs are run."""

import pytest

from exo.scheduler import ExoScheduler
from exo.store.feature_store import FeatureStore


@pytest.mark.asyncio
async def test_late_jobs_still_run(tmp_path):
    # With APScheduler's default 1s grace, a job that fired while another job
    # held the event loop was silently dropped (~25% of GDELT cycles).
    sched = ExoScheduler(store=FeatureStore(data_dir=tmp_path / "features", redis_url=None))
    sched.setup_jobs()
    sched._scheduler.start(paused=True)
    try:
        job = sched._scheduler.get_job("gdelt_ingest")
        assert job.misfire_grace_time == 300
        assert job.coalesce is True
        assert job.max_instances == 1
    finally:
        sched._scheduler.shutdown(wait=False)
