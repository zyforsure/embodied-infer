"""Optional execution plugins shared by model backends."""

from .vision_batch import VisionBatchPlugin, VisionBatchConfig

__all__ = ["VisionBatchPlugin", "VisionBatchConfig"]
