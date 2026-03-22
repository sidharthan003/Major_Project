from typing import Any


def build_phuman_payload(frame_id: int, timestamp_ms: int, sensor_id: str, tracks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "frame_id": frame_id,
        "timestamp_ms": timestamp_ms,
        "sensor_id": sensor_id,
        "tracks": tracks,
    }