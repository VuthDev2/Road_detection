"""
src/utils/export.py
===================
Data export utilities.
Converts structured detection/tracking data into downloadable formats.
No Streamlit dependency — returns plain Python objects.
"""

from __future__ import annotations

import csv
import io

from src.core.tracker import TrackingState


def build_csv_buffer(state: TrackingState) -> io.StringIO:
    """
    Build an in-memory CSV from a TrackingState's defect log.

    Parameters
    ----------
    state : TrackingState
        Accumulated tracking state from a full video run.

    Returns
    -------
    io.StringIO
        In-memory CSV buffer ready to be passed to st.download_button().

    Example
    -------
    >>> buf = build_csv_buffer(state)
    >>> st.download_button("Export Log", data=buf.getvalue(), ...)
    """
    fieldnames = ["Track ID", "Timestamp (seconds)", "Class"]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()

    for entry in state.defect_log.values():
        writer.writerow({
            "Track ID": entry.track_id,
            "Timestamp (seconds)": entry.timestamp_sec,
            "Class": entry.class_name,
        })

    return buffer
