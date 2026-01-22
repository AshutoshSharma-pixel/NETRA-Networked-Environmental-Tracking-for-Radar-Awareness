"""
visuals/renderer.py

A minimal pygame-based renderer for the Doomsday simulation. It draws a vertical
gradient for the sky and simple circle particles. This file keeps the rendering
concise and beginner friendly.
"""

import pygame
from typing import Tuple, Iterable


class Renderer:
    """Simple renderer that manages a pygame surface and draws sky and particles.

    Usage:
        r = Renderer(800, 600)
        r.clear()
        r.draw_sky((top_color, bottom_color))
        r.draw_particles(particles)
        r.present()
    """

    def __init__(self, width: int = 800, height: int = 600, caption: str = "Doomsday"):
        pygame.init()
        self.width = width
        self.height = height
        self._screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(caption)
        self._clock = pygame.time.Clock()

    def clear(self, color: Tuple[int, int, int] = (0, 0, 0)) -> None:
        self._screen.fill(color)

    def draw_sky(self, gradient: Tuple[Tuple[int, int, int], Tuple[int, int, int]]) -> None:
        """Draw a simple vertical gradient from top_color to bottom_color.

        This is done by drawing horizontal lines. It's simple and works for
        beginner projects. For performance in a real project, consider using
        optimized shaders or precomputed surfaces.
        """
        top_color, bottom_color = gradient
        for y in range(self.height):
            t = y / max(1, self.height - 1)
            r = int(top_color[0] + (bottom_color[0] - top_color[0]) * t)
            g = int(top_color[1] + (bottom_color[1] - top_color[1]) * t)
            b = int(top_color[2] + (bottom_color[2] - top_color[2]) * t)
            pygame.draw.line(self._screen, (r, g, b), (0, y), (self.width, y))

    def draw_particles(self, particles: Iterable) -> None:
        """Draw a collection of particles. Each particle is expected to have x, y, size, and get_color()."""
        for p in particles:
            if not getattr(p, "is_alive", lambda: True)():
                continue
            color = p.get_color()
            # pygame expects an (r,g,b,a) surface for per-pixel alpha; we will draw onto the screen using a temporary surface when alpha < 255
            alpha = color[3]
            radius = max(1, int(p.size))
            if alpha >= 255:
                pygame.draw.circle(self._screen, (color[0], color[1], color[2]), (int(p.x), int(p.y)), radius)
            else:
                surf = pygame.Surface((radius * 2 + 2, radius * 2 + 2), pygame.SRCALPHA)
                pygame.draw.circle(surf, (color[0], color[1], color[2], alpha), (radius + 1, radius + 1), radius)
                self._screen.blit(surf, (int(p.x) - radius - 1, int(p.y) - radius - 1))

    def present(self, fps: int = 60) -> float:
        """Update the display and return the time delta in seconds since last call."""
        pygame.display.flip()
        dt = self._clock.tick(fps) / 1000.0
        return dt

    def shutdown(self) -> None:
        pygame.quit()


# End of visuals/renderer.py