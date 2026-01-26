"""Tests for cloud_utils module (unit tests without GEE)."""

from cubexpress.metadata.sensors import (
    AGGREGATED_SENSORS,
    LANDSAT_COMMON_OPTIONAL,
    S2_BOA_BANDS,
    S2_TOA_BANDS,
    SENSORS,
    SensorConfig,
)
from cubexpress.metadata.tables import _parse_sensor_from_id


class TestSensorsDict:
    """Tests for SENSORS dictionary."""

    def test_s2_exists(self):
        assert "S2" in SENSORS
        assert "S2_TOA" in SENSORS
        assert "S2_BOA" in SENSORS

    def test_landsat_exists(self):
        assert "OLI8" in SENSORS
        assert "TM5" in SENSORS
        assert "MSS1" in SENSORS

    def test_aggregated_sensors(self):
        assert "LANDSAT" in SENSORS
        assert "LANDSAT_TOA" in SENSORS
        assert "LANDSAT_BOA" in SENSORS

    def test_sensor_config_type(self):
        for key, config in SENSORS.items():
            assert isinstance(config, SensorConfig)


class TestAggregatedSensors:
    """Tests for AGGREGATED_SENSORS set."""

    def test_contains_landsat(self):
        assert "LANDSAT" in AGGREGATED_SENSORS
        assert "LANDSAT_TOA" in AGGREGATED_SENSORS
        assert "LANDSAT_BOA" in AGGREGATED_SENSORS

    def test_contains_multispectral(self):
        assert "MULTISPECTRAL_TOA" in AGGREGATED_SENSORS
        assert "MULTISPECTRAL_BOA" in AGGREGATED_SENSORS


class TestParseSensorFromId:
    """Tests for _parse_sensor_from_id function."""

    def test_sentinel2(self):
        result = _parse_sensor_from_id("COPERNICUS/S2_HARMONIZED/image")
        assert result == "S2-MSI"

    def test_landsat8_t1(self):
        result = _parse_sensor_from_id("LANDSAT/LC08/C02/T1/image")
        assert result == "OLI8-T1"

    def test_landsat8_t2(self):
        result = _parse_sensor_from_id("LANDSAT/LC08/C02/T2/image")
        assert result == "OLI8-T2"

    def test_landsat5_tm(self):
        result = _parse_sensor_from_id("LANDSAT/LT05/C02/T1/image")
        assert result == "TM5-T1"

    def test_mss(self):
        result = _parse_sensor_from_id("LANDSAT/LM01/C02/T2/image")
        assert result == "MSS1-T2"

    def test_unknown_returns_none(self):
        result = _parse_sensor_from_id("UNKNOWN/COLLECTION/image")
        assert result is None


class TestS2Bands:
    """Tests for Sentinel-2 band constants."""

    def test_toa_bands_count(self):
        assert len(S2_TOA_BANDS) == 13

    def test_boa_bands_count(self):
        # BOA doesn't have B10 (cirrus)
        assert len(S2_BOA_BANDS) == 12

    def test_b10_in_toa_not_boa(self):
        assert "B10" in S2_TOA_BANDS
        assert "B10" not in S2_BOA_BANDS


class TestLandsatCommonOptional:
    """Tests for LANDSAT_COMMON_OPTIONAL set."""

    def test_contains_cloud_cover_land(self):
        assert "CLOUD_COVER_LAND" in LANDSAT_COMMON_OPTIONAL

    def test_contains_sun_angles(self):
        assert "SUN_AZIMUTH" in LANDSAT_COMMON_OPTIONAL
        assert "SUN_ELEVATION" in LANDSAT_COMMON_OPTIONAL
