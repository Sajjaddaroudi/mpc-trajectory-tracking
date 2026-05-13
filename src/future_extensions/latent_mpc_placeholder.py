"""Future latent-space and uncertainty-aware MPC module."""

from __future__ import annotations


class LatentMPCPlaceholder:
    """Placeholder for learned latent dynamics, uncertainty, and MPC coupling."""

    def solve(self) -> None:
        """Reserve the future latent-space MPC API."""
        raise NotImplementedError("Latent-space MPC is not implemented in the classical baseline.")
