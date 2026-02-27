"""
Dual-tier memory system with learned-prototype similarity index.

Hot tier: live Fractal objects. Fast access, full fidelity. Actively learning.
Cold tier: compressed dicts. Smaller footprint. Archived patterns that can
           be restored when needed.

Similarity search uses each fractal's LEARNED PROTOTYPE — not random
embeddings. The prototypes are meaningful because they were shaped by
every input the fractal has ever processed.

Tiering policy: fractals with low fitness AND low recency move to cold.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from .fractal import Fractal
from .compare import similarity


class Memory:
    """Hot/cold memory with prototype-based similarity search."""

    def __init__(self, hot_capacity: int = 1000, cold_capacity: int = 10000):
        self.hot: Dict[str, Fractal] = {}
        self.cold: Dict[str, dict] = {}
        self.hot_capacity = hot_capacity
        self.cold_capacity = cold_capacity
        # Signature index for FunctionFractal similarity search.
        # Maps fractal_id -> normalized output vector (shape signature).
        self._signatures: Dict[str, np.ndarray] = {}

    # ================================================================
    # STORE / RETRIEVE
    # ================================================================

    def store(self, fractal: Fractal):
        """Add a fractal to hot memory."""
        self.hot[fractal.id] = fractal
        # Auto-index function signatures for pattern library search
        if hasattr(fractal, '_signature') and np.any(fractal._signature != 0):
            self._signatures[fractal.id] = fractal._signature
        if len(self.hot) > self.hot_capacity:
            self._evict()

    def get(self, fractal_id: str) -> Optional[Fractal]:
        """Retrieve by ID. Promotes from cold if necessary."""
        if fractal_id in self.hot:
            return self.hot[fractal_id]
        if fractal_id in self.cold:
            return self._promote(fractal_id)
        return None

    def contains(self, fractal_id: str) -> bool:
        """Check if a fractal exists in either tier."""
        return fractal_id in self.hot or fractal_id in self.cold

    # ================================================================
    # SIMILARITY SEARCH
    # ================================================================

    def find_similar(
        self,
        query: np.ndarray,
        domain: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Tuple[Fractal, float]]:
        """Find top-k fractals whose prototypes are most similar to query.

        Uses actual learned prototypes, not random embeddings.
        Optionally filtered by domain.

        Returns list of (fractal, similarity_score) sorted descending.
        """
        candidates = []
        for frac in self.hot.values():
            if domain is not None and frac.domain != domain:
                continue
            # Conform query to fractal's dimensionality for comparison
            if len(query) >= frac.dim:
                q = query[: frac.dim]
            else:
                q = np.zeros(frac.dim)
                q[: len(query)] = query
            sim = similarity(q, frac.prototype)
            candidates.append((frac, sim))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    def update_signature(self, fractal_id: str, signature: np.ndarray) -> None:
        """Store or update the function signature for a fractal."""
        self._signatures[fractal_id] = signature

    def find_similar_by_signature(
        self,
        query_signature: np.ndarray,
        domain: Optional[str] = None,
        top_k: int = 5,
        min_fitness: float = 0.0,
    ) -> List[Tuple[Fractal, float]]:
        """Find fractals whose function signatures best match the query.

        Unlike find_similar() which uses prototypes, this compares
        function output shapes. Designed for FunctionFractal similarity.

        Args:
            query_signature: A normalized output vector (from compute_signature)
            domain: Optional domain filter
            top_k: Number of results to return
            min_fitness: Minimum fitness threshold for candidates

        Returns:
            List of (fractal, similarity_score) sorted descending.
        """
        candidates = []
        for frac_id, sig in self._signatures.items():
            frac = self.hot.get(frac_id)
            if frac is None:
                continue
            if domain is not None and frac.domain != domain:
                continue
            if frac.metrics.fitness() < min_fitness:
                continue
            sim = similarity(query_signature, sig)
            candidates.append((frac, sim))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    # ================================================================
    # TIERING
    # ================================================================

    def _evict(self):
        """Move lowest-fitness fractals from hot to cold.

        Evicts the bottom 10% by fitness, breaking ties by recency.
        Skips children of active composed fractals to avoid breaking
        hierarchies mid-use.
        """
        # Collect IDs of children that are part of active compositions
        protected = set()
        for f in self.hot.values():
            if f.is_composed:
                for child in f.children:
                    protected.add(child.id)

        scored = [
            (fid, f.metrics.fitness(), f.metrics.last_active)
            for fid, f in self.hot.items()
            if fid not in protected
        ]

        if not scored:
            return

        scored.sort(key=lambda x: (x[1], x[2]))  # lowest fitness, then oldest
        n_evict = max(1, len(scored) // 10)
        for fid, _, _ in scored[:n_evict]:
            self._demote(fid)

    def _demote(self, fractal_id: str):
        """Move a fractal from hot to cold (compress it)."""
        if fractal_id in self.hot:
            fractal = self.hot.pop(fractal_id)
            compressed = fractal.compress()
            self.cold[fractal_id] = compressed
            # Enforce cold capacity
            if len(self.cold) > self.cold_capacity:
                oldest_id = min(
                    self.cold.keys(),
                    key=lambda k: self.cold[k]["metrics"].get(
                        "total_exposures", 0
                    ),
                )
                del self.cold[oldest_id]
                self._signatures.pop(oldest_id, None)

    def _promote(self, fractal_id: str) -> Fractal:
        """Restore a fractal from cold to hot."""
        compressed = self.cold.pop(fractal_id)
        fractal = Fractal.decompress(compressed)
        self.hot[fractal_id] = fractal
        return fractal

    # ================================================================
    # STATS
    # ================================================================

    def stats(self) -> dict:
        """Return memory statistics."""
        return {
            "hot_count": len(self.hot),
            "cold_count": len(self.cold),
            "hot_capacity": self.hot_capacity,
            "cold_capacity": self.cold_capacity,
            "signature_count": len(self._signatures),
            "avg_fitness": (
                float(
                    np.mean([f.metrics.fitness() for f in self.hot.values()])
                )
                if self.hot
                else 0.0
            ),
        }
