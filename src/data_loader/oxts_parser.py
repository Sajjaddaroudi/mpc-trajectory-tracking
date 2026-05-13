"""Parser for KITTI raw OXTS packet text files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class OXTSPacket:
    """Subset of an OXTS packet needed by the MPC baseline."""

    lat: float
    lon: float
    alt: float
    roll: float
    pitch: float
    yaw: float
    vf: float
    vl: float
    vu: float
    ax: float
    ay: float
    az: float


def parse_oxts_file(path: Path) -> OXTSPacket:
    """Parse a single KITTI OXTS packet."""
    values = [float(item) for item in path.read_text(encoding="utf-8").strip().split()]
    if len(values) < 14:
        raise ValueError(f"Expected at least 14 OXTS values in {path}, found {len(values)}")
    return OXTSPacket(
        lat=values[0],
        lon=values[1],
        alt=values[2],
        roll=values[3],
        pitch=values[4],
        yaw=values[5],
        vf=values[8],
        vl=values[9],
        vu=values[10],
        ax=values[11],
        ay=values[12],
        az=values[13],
    )


def load_oxts_packets(oxts_data_dir: Path, max_frames: int | None = None) -> list[OXTSPacket]:
    """Load ordered KITTI OXTS packets from an `oxts/data` directory."""
    packet_files = sorted(oxts_data_dir.glob("*.txt"))
    if max_frames is not None:
        packet_files = packet_files[:max_frames]
    if not packet_files:
        raise FileNotFoundError(f"No OXTS packet files found in {oxts_data_dir}")
    return [parse_oxts_file(path) for path in packet_files]


def packets_to_arrays(packets: list[OXTSPacket]) -> dict[str, np.ndarray]:
    """Convert packet dataclasses to NumPy arrays."""
    fields = OXTSPacket.__dataclass_fields__.keys()
    return {field: np.asarray([getattr(packet, field) for packet in packets], dtype=float) for field in fields}
