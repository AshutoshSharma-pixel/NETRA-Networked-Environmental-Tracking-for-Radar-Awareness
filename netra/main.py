"""
main.py

A beginner-friendly radar-style airspace monitoring simulation using pygame.
This renders a circular radar scope, a rotating sweep, moving targets that
bounce inside the radar area, and labels showing simple target IDs.

No weapons logic. Visualization and tracking only.
"""

import sys
import math
import random
import pygame
from typing import List, Tuple

# Simple color palette
BG_COLOR = (6, 10, 12)
RADAR_GREEN = (50, 255, 80)
DARK_GREEN = (10, 40, 12)
SWEEP_ALPHA = 40


class Target:
    """Represents a moving target inside the radar circle."""

    def __init__(self, tid: str, x: float, y: float, vx: float, vy: float, size: int = 5):
        self.id = tid
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.size = size
        self.color = RADAR_GREEN

    def update(self, dt: float, center: Tuple[int, int], radius: float) -> None:
        # Move
        self.x += self.vx * dt
        self.y += self.vy * dt

        # Keep inside the circular radar area by reflecting velocity at the boundary
        cx, cy = center
        dx = self.x - cx
        dy = self.y - cy
        dist = math.hypot(dx, dy)
        margin = self.size + 2
        if dist + margin >= radius:
            # Normal vector from center to point
            if dist == 0:
                nx, ny = 1.0, 0.0
            else:
                nx, ny = dx / dist, dy / dist
            # Reflect velocity: v' = v - 2*(v·n)*n
            vdotn = self.vx * nx + self.vy * ny
            self.vx = self.vx - 2 * vdotn * nx
            self.vy = self.vy - 2 * vdotn * ny
            # Nudge inside so we don't stick to the edge
            self.x = cx + nx * (radius - margin)
            self.y = cy + ny * (radius - margin)

    def screen_pos(self) -> Tuple[int, int]:
        return int(self.x), int(self.y)


class Radar:
    """Handles drawing the radar scope and sweep."""

    def __init__(self, surface: pygame.Surface, center: Tuple[int, int], radius: int):
        self.surface = surface
        self.center = center
        self.radius = radius
        self.sweep_angle = 0.0  # radians
        self.sweep_speed = math.radians(90)  # 90 degrees per second
        self.sweep_width = math.radians(2.5)  # narrow sweep wedge
        self.font = pygame.font.SysFont("consolas", 14)

    def update(self, dt: float) -> None:
        self.sweep_angle = (self.sweep_angle + self.sweep_speed * dt) % (math.pi * 2)

    def draw_scope(self) -> None:
        cx, cy = self.center
        # Fill background circle area slightly darker to emphasize scope
        pygame.draw.circle(self.surface, DARK_GREEN, (cx, cy), self.radius)

        # Concentric rings
        ring_count = 4
        for i in range(1, ring_count + 1):
            r = int(self.radius * i / (ring_count + 1))
            pygame.draw.circle(self.surface, RADAR_GREEN, (cx, cy), r, 1)

        # Crosshair lines
        pygame.draw.line(self.surface, RADAR_GREEN, (cx - self.radius, cy), (cx + self.radius, cy), 1)
        pygame.draw.line(self.surface, RADAR_GREEN, (cx, cy - self.radius), (cx, cy + self.radius), 1)

    def draw_sweep(self) -> None:
        cx, cy = self.center
        # Create a temporary surface with alpha for the sweep wedge
        sweep_surf = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
        sw_cx, sw_cy = self.radius, self.radius

        # Build points for the wedge polygon
        points = [(sw_cx, sw_cy)]
        step = max(1, int(self.sweep_width * self.radius / 0.04))
        half = self.sweep_width / 2.0
        start_ang = self.sweep_angle - half
        end_ang = self.sweep_angle + half
        steps = max(4, int((end_ang - start_ang) / (math.radians(0.5))))
        for i in range(steps + 1):
            a = start_ang + (end_ang - start_ang) * (i / max(1, steps))
            px = sw_cx + math.cos(a) * self.radius
            py = sw_cy + math.sin(a) * self.radius
            points.append((px, py))

        # Draw a translucent wedge
        pygame.draw.polygon(sweep_surf, (RADAR_GREEN[0], RADAR_GREEN[1], RADAR_GREEN[2], SWEEP_ALPHA), points)

        # Add a bright line at the leading edge of the sweep
        lead_x = sw_cx + math.cos(self.sweep_angle) * self.radius
        lead_y = sw_cy + math.sin(self.sweep_angle) * self.radius
        pygame.draw.line(sweep_surf, (RADAR_GREEN[0], RADAR_GREEN[1], RADAR_GREEN[2], 200), (sw_cx, sw_cy), (lead_x, lead_y), 2)

        # Blit the sweep surface centered on the main surface
        self.surface.blit(sweep_surf, (cx - self.radius, cy - self.radius), special_flags=pygame.BLEND_PREMULTIPLIED)

    def draw_targets(self, targets: List[Target]) -> None:
        cx, cy = self.center
        for t in targets:
            tx, ty = t.screen_pos()
            # Draw the target dot
            pygame.draw.circle(self.surface, t.color, (tx, ty), t.size)
            # Draw ID slightly offset to avoid overlap
            id_surf = self.font.render(t.id, True, RADAR_GREEN)
            offset_x = 8
            offset_y = -8
            self.surface.blit(id_surf, (tx + offset_x, ty + offset_y))

    def draw(self, targets: List[Target]) -> None:
        # Draw scope elements and sweep then targets
        self.draw_scope()
        self.draw_sweep()
        self.draw_targets(targets)


def random_target(center: Tuple[int, int], radius: int, tid: str) -> Target:
    cx, cy = center
    # Pick a random position inside the circle
    while True:
        rx = random.uniform(cx - radius, cx + radius)
        ry = random.uniform(cy - radius, cy + radius)
        if (rx - cx) ** 2 + (ry - cy) ** 2 <= (radius - 10) ** 2:
            break
    angle = random.uniform(0, math.pi * 2)
    speed = random.uniform(20, 80)
    vx = math.cos(angle) * speed
    vy = math.sin(angle) * speed
    return Target(tid, rx, ry, vx, vy, size=random.randint(3, 6))


def main() -> None:
    pygame.init()
    WIDTH, HEIGHT = 900, 900
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Airspace Monitoring - Radar Simulation")
    clock = pygame.time.Clock()

    center = (WIDTH // 2, HEIGHT // 2)
    radius = min(WIDTH, HEIGHT) // 2 - 20

    # Create radar
    radar = Radar(screen, center, radius)

    # Create some initial targets
    targets: List[Target] = []
    for i in range(6):
        targets.append(random_target(center, radius, f"T{i+1}"))

    running = True
    while running:
        dt = clock.tick(60) / 1000.0  # delta seconds (60 FPS cap)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Add a new target at mouse position with a random velocity
                mx, my = event.pos
                # Only add if inside radar circle
                dx = mx - center[0]
                dy = my - center[1]
                if dx * dx + dy * dy <= radius * radius:
                    tid = f"T{len(targets) + 1}"
                    angle = random.uniform(0, math.pi * 2)
                    speed = random.uniform(20, 80)
                    vx = math.cos(angle) * speed
                    vy = math.sin(angle) * speed
                    targets.append(Target(tid, mx, my, vx, vy, size=random.randint(3, 6)))

        # Update simulation
        radar.update(dt)
        for t in targets:
            t.update(dt, center, radius)

        # Clear screen
        screen.fill(BG_COLOR)

        # Draw radar and targets
        radar.draw(targets)

        # Draw a faint label
        font = pygame.font.SysFont("consolas", 16)
        label = font.render("Airspace Monitoring - Visualization Only", True, (100, 180, 120))
        screen.blit(label, (10, 10))

        pygame.display.flip()

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
