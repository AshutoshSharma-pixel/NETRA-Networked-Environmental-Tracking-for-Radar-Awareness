"""
engine/particle.py

A lightweight Particle class for Doomsday simulation. Handles basic physics,
aging, fade-out, and simple turbulence. Designed to be used by the simulation
and rendered by the renderer.
"""

from typing import Tuple
import math
import random

Color = Tuple[int, int, int, int]  # RGBA


class Particle:
    """Simple particle with position, velocity, gravity, turbulence, and fading.

    Attributes:
        x, y: position
        vx, vy: velocity
        ax, ay: acceleration (external)
        life: remaining life (seconds)
        lifespan: initial life (seconds)
        color: base RGBA color
        size: visual size (radius)
    """

    def __init__(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0, *,
                 lifespan: float = 2.0, color: Color = (255, 255, 255, 255), size: float = 3.0):
        self.x = float(x)
        self.y = float(y)
        self.vx = float(vx)
        self.vy = float(vy)
        self.ax = 0.0
        self.ay = 0.0

        self.lifespan = max(0.0001, float(lifespan))
        self.life = self.lifespan
        self.color = color
        self.size = float(size)

        # Local random seed for per-particle turbulence
        self._seed = random.random() * 1000.0

    def apply_force(self, fx: float, fy: float) -> None:
        """Apply an instantaneous force (acceleration) to the particle."""
        self.ax += fx
        self.ay += fy

    def update(self, dt: float, gravity: float = 300.0, turbulence: float = 20.0) -> None:
        """Advance particle state by dt seconds.

        gravity: pixels per second squared downward.
        turbulence: magnitude of per-frame velocity jitter.
        """
        if self.life <= 0.0:
            return

        # Simple gravity accumulation
        self.ay += gravity * dt

        # Turbulence as a noise-like jitter based on time and seed
        t = max(0.0, min(1.0, (self.lifespan - self.life) / self.lifespan))
        jitter_x = (math.sin((self._seed + t * 10.0) * 12.9898) * 43758.5453) % 1.0
        jitter_y = (math.cos((self._seed + t * 8.0) * 78.233) * 12345.6789) % 1.0
        jitter_x = (jitter_x - 0.5) * 2.0 * turbulence
        jitter_y = (jitter_y - 0.5) * 2.0 * turbulence

        # Integrate velocity
        self.vx += (self.ax + jitter_x) * dt
        self.vy += (self.ay + jitter_y) * dt

        # Integrate position
        self.x += self.vx * dt
        self.y += self.vy * dt

        # Reset accumulators
        self.ax = 0.0
        self.ay = 0.0

        # Age the particle
        self.life -= dt
        if self.life < 0.0:
            self.life = 0.0

    def is_alive(self) -> bool:
        return self.life > 0.0

    def get_alpha(self) -> int:
        """Return current alpha (0-255) based on remaining life."""
        frac = max(0.0, min(1.0, self.life / self.lifespan))
        return int(self.color[3] * frac)

    def get_color(self) -> Tuple[int, int, int, int]:
        """Return color with current alpha applied."""
        a = self.get_alpha()
        return (self.color[0], self.color[1], self.color[2], a)


# End of engine/particle.py
