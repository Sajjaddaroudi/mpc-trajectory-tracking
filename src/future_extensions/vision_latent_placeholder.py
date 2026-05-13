"""Future camera latent encoding module.

This file intentionally contains interface notes rather than an implementation.
The planned extension is to encode KITTI `image_02` frames into compact latent
state variables that can augment or replace hand-engineered trajectory features.
"""

from __future__ import annotations


class VisionLatentEncoderPlaceholder:
    """Placeholder interface for a future camera representation model."""

    def encode(self, image_path: str) -> None:
        """Reserve the future image-to-latent API."""
        raise NotImplementedError("Camera latent encoding is a planned research extension.")
