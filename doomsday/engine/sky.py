"""
engine/sky.py

Sky class that manages the sky color, smooth transitions between colors,
and simple pulse effects. This is intentionally kept simple and beginner-friendly.
"""

from typing import Tuple, Optional

Color = Tuple[int, int, int]


def _lerp_color(a: Color, b: Color, t: float) -> Color:
    """Linearly interpolate between two RGB colors."""
    t = max(0.0, min(1.0, t))
    return (
        int(a[0] + (b[0] - a[0]) * t),
        int(a[1] + (b[1] - a[1]) * t),
        int(a[2] + (b[2] - a[2]) * t),
    )


class Sky:
    """A simple sky manager.

    Responsibilities:
    - Hold a base color and an optional target color for smooth transitions.
    - Advance transitions over time (update(dt)).
    - Provide a two-color vertical gradient (top/bottom) for rendering.
    - Support brief color pulses.
    """

    def __init__(self, top_color: Color = (20, 30, 80), bottom_color: Color = (100, 110, 160)):
        # Base gradient colors
        self.top_color: Color = top_color
        self.bottom_color: Color = bottom_color

        # Transition targets and timing
        self._target_top: Optional[Color] = None
        self._target_bottom: Optional[Color] = None
        self._transition_duration: float = 0.0
        self._transition_elapsed: float = 0.0

        # Pulse effect (temporary overlay transition)
        self._pulse_top: Optional[Color] = None
        self._pulse_bottom: Optional[Color] = None
        self._pulse_duration: float = 0.0
        self._pulse_elapsed: float = 0.0

    def set_target(self, top: Color, bottom: Color, duration: float = 3.0) -> None:
        """Begin a smooth transition from the current gradient to target colors.

        duration is in seconds.
        """
        self._target_top = top
        self._target_bottom = bottom
        self._transition_duration = max(0.0001, duration)
        self._transition_elapsed = 0.0

    def trigger_pulse(self, top: Color, bottom: Color, duration: float = 0.6) -> None:
        """Trigger a short pulse (temporary color overlay)."""
        self._pulse_top = top
        self._pulse_bottom = bottom
        self._pulse_duration = max(0.0001, duration)
        self._pulse_elapsed = 0.0

    def update(self, dt: float) -> None:
        """Advance transitions by dt seconds. Call this every frame from the simulation."""
        # Update main transition
        if self._target_top is not None and self._target_bottom is not None:
            self._transition_elapsed += dt
            t = min(1.0, self._transition_elapsed / self._transition_duration)
            # Smoothstep easing for nicer visuals
            t_eased = t * t * (3 - 2 * t)
            self.top_color = _lerp_color(self.top_color, self._target_top, t_eased)
            self.bottom_color = _lerp_color(self.bottom_color, self._target_bottom, t_eased)
            if t >= 1.0:
                # Finish transition
                self._target_top = None
                self._target_bottom = None

        # Update pulse (applies on top of base gradient)
        if self._pulse_top is not None and self._pulse_bottom is not None:
            self._pulse_elapsed += dt
            pt = min(1.0, self._pulse_elapsed / self._pulse_duration)
            # Pulse fades in then out (triangle)
            if pt < 0.5:
                amp = (pt / 0.5)
            else:
                amp = max(0.0, (1.0 - (pt - 0.5) / 0.5))
            # Blend base -> pulse by amp
            blended_top = _lerp_color(self.top_color, self._pulse_top, amp)
            blended_bottom = _lerp_color(self.bottom_color, self._pulse_bottom, amp)

            # If pulse ended, clear it
            if self._pulse_elapsed >= self._pulse_duration:
                self._pulse_top = None
                self._pulse_bottom = None
                self._pulse_elapsed = 0.0

            # Expose a temporary gradient during the pulse by saving to _last_pulse
            self._last_pulse = (blended_top, blended_bottom)
        else:
            self._last_pulse = None

    def get_gradient(self) -> Tuple[Color, Color]:
        """Return (top_color, bottom_color) to be used by the renderer.

        If a pulse is active, returns the pulsed blended colors.
        """
        if getattr(self, "_last_pulse", None) is not None:
            return self._last_pulse  # type: ignore
        return self.top_color, self.bottom_color

    def instant_set(self, top: Color, bottom: Color) -> None:
        """Immediately set the base gradient colors (cancel transitions)."""
        self.top_color = top
        self.bottom_color = bottom
        self._target_top = None
        self._target_bottom = None
        self._transition_duration = 0.0
        self._transition_elapsed = 0.0


# End of engine/sky.py
