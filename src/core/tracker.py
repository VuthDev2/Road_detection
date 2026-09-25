"""
src/core/tracker.py
===================
Parses ByteTrack output from Ultralytics Results into a structured
DefectLog — a dict of unique track IDs to their first-seen metadata.

This module is pure Python with no UI dependency; it can be used
from Streamlit, a CLI script, or unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ultralytics.engine.results import Results

from config.settings import DAMAGE_CLASSES


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DefectEntry:
    """A single unique defect detected and tracked in a video."""
    track_id: int
    timestamp_sec: float
    class_name: str


@dataclass
class TrackingState:
    """
    Accumulates tracking data across video frames.

    Attributes
    ----------
    defect_log : dict[int, DefectEntry]
        Maps track ID → first-seen DefectEntry.
    unique_ids : set[int]
        All track IDs seen so far.
    class_ids : dict[str, set[int]]
        Maps class name → set of track IDs for that class.
    raw_detections : int
        Total bounding boxes seen (may include duplicate tracks).
    """
    defect_log: dict[int, DefectEntry] = field(default_factory=dict)
    unique_ids: set[int] = field(default_factory=set)
    class_ids: dict[str, set[int]] = field(
        default_factory=lambda: {cls: set() for cls in DAMAGE_CLASSES}
    )
    raw_detections: int = 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class TrackerParser:
    """
    Stateless helper that updates a TrackingState from a single
    Ultralytics Results object (one frame of tracked output).

    Usage
    -----
    >>> state = TrackingState()
    >>> parser = TrackerParser()
    >>> for frame in video:
    ...     results = detector.track(frame, cfg)
    ...     parser.update(state, results[0], timestamp_sec)
    """

    @staticmethod
    def update(
        state: TrackingState,
        result: "Results",
        timestamp_sec: float,
    ) -> None:
        """
        Update the shared TrackingState with detections from one frame.

        Parameters
        ----------
        state : TrackingState
            Mutable state object accumulating data across frames.
        result : Results
            Single-frame result from detector.track().
        timestamp_sec : float
            Time position of this frame in the video (seconds).
        """
        if result.boxes.cls is None:
            return

        # Count all raw bounding boxes (untracked frames included)
        state.raw_detections += len(result.boxes.cls)

        if result.boxes.id is None:
            return  # Tracking failed for this frame — skip

        track_ids = result.boxes.id.int().cpu().tolist()
        classes = result.boxes.cls.int().cpu().tolist()
        class_names = result.names

        for tid, cls_idx in zip(track_ids, classes):
            class_name = class_names[cls_idx]
            state.unique_ids.add(tid)

            if class_name in state.class_ids:
                state.class_ids[class_name].add(tid)

            # Only log first occurrence of each unique ID
            if tid not in state.defect_log:
                state.defect_log[tid] = DefectEntry(
                    track_id=tid,
                    timestamp_sec=round(timestamp_sec, 3),
                    class_name=class_name,
                )

    @staticmethod
    def class_counts(state: TrackingState) -> dict[str, int]:
        """Return a dict of class_name → number of unique tracked IDs."""
        return {cls: len(ids) for cls, ids in state.class_ids.items()}
