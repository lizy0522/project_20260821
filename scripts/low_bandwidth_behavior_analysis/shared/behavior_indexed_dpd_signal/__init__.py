"""Signal-domain primitives for behavior-indexed DPD experiments."""

from .low_bandwidth_observation import (
    LowBandwidthObservationBank,
    LowBandwidthObservationOperator,
    SampleRateSpec,
)

__all__ = [
    "LowBandwidthObservationBank",
    "LowBandwidthObservationOperator",
    "SampleRateSpec",
]
