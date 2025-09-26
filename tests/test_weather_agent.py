import sys
import types
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from weather_agent import (  # noqa: E402  # isort:skip
    SIMULATED_DATASET,
    WeatherDataset,
    iter_available_locations,
    query_weather,
)


def test_dataset_lookup_known_city():
    assert SIMULATED_DATASET.lookup("北京").startswith("北京")


def test_dataset_lookup_default_fallback():
    result = SIMULATED_DATASET.lookup("不存在的城市")
    assert "不存在的城市" in result


def test_iter_available_locations():
    locations = set(iter_available_locations())
    assert {"北京", "上海"}.issubset(locations)


def test_query_weather_without_sdk_falls_back():
    class DummyClient(types.SimpleNamespace):
        pass

    result = query_weather("深圳", client=DummyClient())
    assert result == SIMULATED_DATASET.lookup("深圳")


def test_weather_dataset_rejects_empty_location():
    dataset = WeatherDataset({})
    with pytest.raises(ValueError):
        dataset.lookup("   ")
