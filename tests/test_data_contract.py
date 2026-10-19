import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = (
    "weather_data.csv",
    "soil_data.csv",
    "crop_data.csv",
    "iot_sensor_data.csv",
)


def test_required_sample_datasets_have_headers_and_rows() -> None:
    for filename in DATASETS:
        with (ROOT / "data" / filename).open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
        assert reader.fieldnames
        assert len(reader.fieldnames) == len(set(reader.fieldnames))
        assert rows
