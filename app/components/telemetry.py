"""
app/components/telemetry.py
===========================
Live telemetry panel rendered in the right column.
Accepts counts and updates Streamlit metric placeholders — no inference logic.
"""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from config.settings import DAMAGE_CLASSES


@dataclass
class TelemetryWidgets:
    """
    Holds references to Streamlit placeholder widgets for each damage class
    plus a total counter.  Created once; updated every frame.
    """
    class_placeholders: dict[str, st.delta_generator.DeltaGenerator]
    total_placeholder: st.delta_generator.DeltaGenerator


def render_telemetry_panel() -> TelemetryWidgets:
    """
    Render the 'Live Telemetry' header and empty metric placeholders.

    Returns
    -------
    TelemetryWidgets
        References to the metric placeholders so callers can update them.
    """
    st.subheader("📡 Live Telemetry")
    st.caption("Unique tracked defects detected in the footage.")
    st.markdown("---")

    placeholders: dict[str, st.delta_generator.DeltaGenerator] = {}
    for cls_name in DAMAGE_CLASSES:
        placeholders[cls_name] = st.empty()

    st.markdown("---")
    total_placeholder = st.empty()

    return TelemetryWidgets(
        class_placeholders=placeholders,
        total_placeholder=total_placeholder,
    )


def update_telemetry(
    widgets: TelemetryWidgets,
    class_counts: dict[str, int],
    total_label: str,
    total_value: str | int,
) -> None:
    """
    Push new counts into the telemetry widget placeholders.

    Parameters
    ----------
    widgets : TelemetryWidgets
        The widget references returned by render_telemetry_panel().
    class_counts : dict[str, int]
        Mapping of class name → count to display.
    total_label : str
        Label for the total metric (e.g. "Total Defects").
    total_value : str | int
        Value for the total metric.
    """
    for cls_name, placeholder in widgets.class_placeholders.items():
        placeholder.metric(cls_name, class_counts.get(cls_name, 0))

    widgets.total_placeholder.metric(total_label, total_value)
