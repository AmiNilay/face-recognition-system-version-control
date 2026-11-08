"""
Spatial Vector Indexing & Accelerated Nearest-Neighbor Search Engine
Module: src/vector_index.py

This module implements O(log N) approximate and exact nearest-neighbor search
for 128-dimensional facial embedding vectors using scipy.spatial.cKDTree.

Beginner Concept:
- In linear brute-force matching, comparing a newly detected face against N enrolled
  faces requires computing Euclidean distance N times (O(N) complexity).
- When a company or campus has hundreds or thousands of enrolled users, linear scan
  causes latency and dropped video frames.
- A k-d tree (k-dimensional tree) partitions 128-dimensional space into a binary
  search tree, pruning distant branches and identifying closest matches in O(log N) time.
- If scipy is not available or if the index contains fewer than 2 vectors, the engine
  gracefully falls back to optimized NumPy matrix vectorization.

All timestamps follow Indian Standard Time (IST, UTC+05:30).
Zero emojis across all code, HUD readouts, and logs. Developer: Nilay Naha.
"""

import numpy as np
from typing import List, Tuple, Optional, Any

try:
    from scipy.spatial import cKDTree
    HAS_CKDTREE = True
except ImportError:
    HAS_CKDTREE = False


class VectorIndex:
    """
    High-performance vector index for 128-dimensional biometric embeddings.
    """

    def __init__(self, use_kdtree: bool = True):
        """
        Initialize the spatial vector search engine.

        Args:
            use_kdtree (bool): Whether to build and query a cKDTree spatial index.
        """
        self.use_kdtree: bool = bool(use_kdtree and HAS_CKDTREE)
        self.tree: Optional[Any] = None
        self.encodings: List[np.ndarray] = []
        self.names: List[str] = []

    def build(self, encodings: List[np.ndarray], names: List[str]) -> bool:
        """
        Build or rebuild the spatial kd-tree index from enrolled vectors.

        Args:
            encodings (list): List of 128-dimensional numpy arrays.
            names (list): Corresponding human identity labels.

        Returns:
            bool: True if index was successfully built with cKDTree; False if fallback mode.
        """
        self.encodings = list(encodings)
        self.names = list(names)

        if not self.encodings or not self.use_kdtree or len(self.encodings) < 2:
            self.tree = None
            return False

        try:
            matrix = np.array(self.encodings, dtype=np.float64)
            if matrix.ndim == 2 and matrix.shape[0] >= 2 and matrix.shape[1] == 128:
                self.tree = cKDTree(matrix)
                return True
            else:
                self.tree = None
                return False
        except Exception:
            self.tree = None
            return False

    def query(
        self,
        query_encoding: np.ndarray,
        tolerance: float = 0.6
    ) -> Tuple[str, float, float]:
        """
        Find the closest enrolled identity for a query 128-d feature vector.

        Args:
            query_encoding (np.ndarray): 128-dimensional query vector.
            tolerance (float): Maximum Euclidean distance threshold. Default 0.6.

        Returns:
            tuple: (matched_name, confidence_percentage, euclidean_distance)
                   Returns ("Unknown", 0.0, distance) if no match meets tolerance.
        """
        if not self.encodings or not self.names or query_encoding is None:
            return "Unknown", 0.0, 1.0

        query_vec = np.array(query_encoding, dtype=np.float64).flatten()
        if len(query_vec) != 128:
            return "Unknown", 0.0, 1.0

        # Strategy A: Accelerated O(log N) kd-tree search
        if self.tree is not None:
            try:
                dist, idx = self.tree.query(query_vec, k=1)
                dist = float(dist)
                idx = int(idx)
            except Exception:
                dist, idx = self._linear_scan(query_vec)
        else:
            # Strategy B: Optimized vectorized NumPy Euclidean distance
            dist, idx = self._linear_scan(query_vec)

        # Convert distance into an intuitive confidence percentage
        confidence = max(0.0, (1.0 - dist)) * 100.0

        if dist <= tolerance and 0 <= idx < len(self.names):
            return self.names[idx], confidence, dist
        else:
            return "Unknown", 0.0, dist

    def _linear_scan(self, query_vec: np.ndarray) -> Tuple[float, int]:
        """
        Vectorized fallback Euclidean distance search over all enrolled vectors.

        Args:
            query_vec (np.ndarray): 128-d float64 query array.

        Returns:
            tuple: (min_distance, best_match_index)
        """
        matrix = np.array(self.encodings, dtype=np.float64)
        diff = matrix - query_vec
        distances = np.linalg.norm(diff, axis=1)
        best_idx = int(np.argmin(distances))
        min_dist = float(distances[best_idx])
        return min_dist, best_idx

    def clear(self):
        """Reset and empty the vector index."""
        self.encodings.clear()
        self.names.clear()
        self.tree = None

    def __len__(self) -> int:
        """Return the count of indexed vectors."""
        return len(self.encodings)
