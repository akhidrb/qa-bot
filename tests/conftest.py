import os
import pytest


@pytest.fixture(autouse=True)
def _env_openai_key(monkeypatch):
    # prevent config from failing fast in tests
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    # if your get_settings uses @lru_cache, clear it between tests
    try:
        from app.core.config import get_settings
        get_settings.cache_clear()
    except Exception:
        pass
    yield
    try:
        from app.core.config import get_settings
        get_settings.cache_clear()
    except Exception:
        pass
