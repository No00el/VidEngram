"""Shared test fixtures."""
import pytest
from pathlib import Path


@pytest.fixture
def tmp_work_dir(tmp_path):
    """Provide a temporary work directory."""
    work = tmp_path / "videngram_test"
    work.mkdir()
    return work


@pytest.fixture
def sample_config(tmp_work_dir):
    """Config pointing at temp directory."""
    from videngram.config import VidEngramConfig
    return VidEngramConfig(work_dir=tmp_work_dir)
