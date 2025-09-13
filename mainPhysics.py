import sys
from typing import Tuple

import numpy as np
import pygame
import pygame_widgets
from pygame.font import SysFont
from pygame_widgets.slider import Slider
from pygame_widgets.textbox import TextBox

# -----------------------------
# Константы и параметры
# -----------------------------
N = 15
W0 = 0.5
SIGMA = 0.3

WIDTH = 1500
HEIGHT = 800
FPS = 60

# Графики: Sync
SYNC_VALUES = []
SYNC_GRAPH_X = 10
SYNC_GRAPH_Y = 50
SYNC_GRAPH_SCALE = 200
SYNC_GRAPH_WIDTH = 250
SYNC_GRAPH_X_OFFSET = 20

# Графики: число «горящих» светлячков
LIT_VALUES = []
COMMON_GRAPH_X = 10
COMMON_GRAPH_Y = 50
COMMON_GRAPH_SCALE = 200
COMMON_GRAPH_WIDTH = 250
COMMON_GRAPH_X_OFFSET = 20

TICK_FONT_SIZE = 12

# Порог фиксации синхронизации
SYNC_THRESHOLD = 0.99

pygame.init()
FONT = pygame.font.Font(None, 36)
TICK_FONT = SysFont("Arial", TICK_FONT_SIZE)


# -----------------------------
# Вспомогательные функции
# -----------------------------
def oscillators(n: int = N, w0: float = W0, sigma: float = SIGMA) -> np.ndarray:
    """Собственные частоты осцилляторов."""
    return w0 + sigma * np.random.uniform(-1, 1, n)


def add_outer_glow(image: pygame.Surface, color=(0, 191, 255), size: int = 4) -> pygame.Surface:
    """Простая «светящаяся» обводка для частицы."""
    new_image = image.copy()
    for r in range(size):
        temp_image = new_image.copy()
        temp_image.fill(0)
        pygame.draw.circle(temp_image, color, (12, 12), 12 - r)
        new_image.blit(temp_image, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)
    return new_image


def lerp_color(c1: Tuple[int, int, int], c2: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    """Линейная интерполяция цветов."""
    return (
        int(c1[0] * (1 - t) + c2[0] * t),
        int(c1[1] * (1 - t) + c2[1] * t),
        int(c1[2] * (1 - t) + c2[2] * t),
    )


def kuramoto_dynamics(phases: np.ndarray, freq: np.ndarray, k: float) -> np.ndarray:
    """Правая часть системы Курамото."""
    # Векторизованная форма: для каждого i считаем сумму синусов
    # sin_sum_i = sum_j sin(theta_j - theta_i)
    delta = phases[None, :] - phases[:, None]  # (N, N)
    sin_sum = np.sum(np.sin(delta), axis=1)
    return freq + (k / len(phases)) * sin_sum


def plot_sync(phases: np.ndarray, surface: pygame.Surface) -> float:
    """Обновление и отрисовка графика Sync. Возвращает текущее значение Sync."""
    # Параметр порядка R в [0, 1]; для удобства отображения масштабируем в [0, 1]
    r = np.abs(np.mean(np.exp(1j * phases)))
    sync = (r + 1.0) / 2.0
    SYNC_VALUES.append(sync)
    if len(SYNC_VALUES) > SYNC_GRAPH_WIDTH:
        SYNC_VALUES.pop(0)

    # Подпись
    sync_label = FONT.render(f"Sync: {sync:.3f}", True, (255, 255, 255))
    surface.blit(sync_label, (10, 10))

    # Оси
    pygame.draw.line(
        surface,
        (255, 255, 255),
        (SYNC_GRAPH_X + SYNC_GRAPH_X_OFFSET, SYNC_GRAPH_Y),
        (SYNC_GRAPH_X + SYNC_GRAPH_X_OFFSET, SYNC_GRAPH_Y + SYNC_GRAPH_SCALE),
        1,
    )
    pygame.draw.line(
        surface,
        (255, 255, 255),
        (SYNC_GRAPH_X + SYNC_GRAPH_X_OFFSET, SYNC_GRAPH_Y + SYNC_GRAPH_SCALE),
        (SYNC_GRAPH_X + SYNC_GRAPH_X_OFFSET + SYNC_GRAPH_WIDTH, SYNC_GRAPH_Y + SYNC_GRAPH_SCALE),
        1,
    )

    # Засечки и подписи
    step = 0.1
    y_ticks = int(1.0 / step)
    for i in range(1, y_ticks):
        y = SYNC_GRAPH_Y + SYNC_GRAPH_SCALE - i * step * SYNC_GRAPH_SCALE
        pygame.draw.line(
            surface,
            (255, 255, 255),
            (SYNC_GRAPH_X + SYNC_GRAPH_X_OFFSET - 5, y),
            (SYNC_GRAPH_X + SYNC_GRAPH_X_OFFSET, y),
            1,
        )
        tick_label = TICK_FONT.render(f"{i * step:.1f}", True, (255, 255, 255))
        surface.blit(tick_label, (SYNC_GRAPH_X + SYNC_GRAPH_X_OFFSET - 30, y - 10))

    # Линия графика
    if SYNC_VALUES:
        max_val = max(SYNC_VALUES)
        max_val = max(max_val, 1e-6)  # защита от деления на ноль
        for i in range(1, len(SYNC_VALUES)):
            t = SYNC_VALUES[i] / max_val
            color = lerp_color((255, 0, 0), (0, 0, 255), t)
            x1 = SYNC_GRAPH_X + SYNC_GRAPH_X_OFFSET + i - 1
            y1 = SYNC_GRAPH_Y + SYNC_GRAPH_SCALE - SYNC_VALUES[i - 1] * SYNC_GRAPH_SCALE
            x2 = SYNC_GRAPH_X + SYNC_GRAPH_X_OFFSET + i
            y2 = SYNC_GRAPH_Y + SYNC_GRAPH_SCALE - SYNC_VALUES[i] * SYNC_GRAPH_SCALE
            pygame.draw.line(surface, color, (x1, y1), (x2, y2), 3)
    return sync


def plot_lit_count(phases: np.ndarray, surface: pygame.Surface, origin_y: int) -> None:
    """График количества «горящих» светлячков."""
    lit = int(np.sum(np.sin(phases) > 0))
    LIT_VALUES.append(lit)
    if len(LIT_VALUES) > COMMON_GRAPH_WIDTH:
        LIT_VALUES.pop(0)

    label = FONT.render(f"Lit fireflies: {lit:d}", True, (255, 255, 255))
    surface.blit(label, (10, origin_y - 45))

    # Оси
    pygame.draw.line(surface, (255, 255, 255), (20, 500), (20, 720), 1)
    pygame.draw.line(surface, (255, 255, 255), (20, 720), (280, 720), 1)

    # Линия графика
    for i in range(1, len(LIT_VALUES)):
        x1 = COMMON_GRAPH_X + COMMON_GRAPH_X_OFFSET + i - 1
        y1 = origin_y + COMMON_GRAPH_SCALE - LIT_VALUES[i - 1] * COMMON_GRAPH_SCALE / N
        x2 = COMMON_GRAPH_X + COMMON_GRAPH_X_OFFSET + i
        y2 = origin_y + COMMON_GRAPH_SCALE - LIT_VALUES[i] * COMMON_GRAPH_SCALE / N
        pygame.draw.line(surface, (0, 255, 0), (x1, y1), (x2, y2), 1)


def plot_scene(phases: np.ndarray, surface: pygame.Surface) -> None:
    """Полная сцена: графики и анимация частиц."""
    # График Sync
    plot_sync(phases, surface)

    # График «горящих»
    plot_lit_count(phases, surface, COMMON_GRAPH_Y + COMMON_GRAPH_SCALE + 250)

    # Частицы на окружности
    radius = 100
    for i in range(len(phases)):
        x0 = WIDTH / 2 + radius * np.cos(i / len(phases) * 2 * np.pi)
        y0 = HEIGHT / 2 + radius * np.sin(i / len(phases) * 2 * np.pi)

        angle = (i / len(phases) * 2 * np.pi) + np.pi / 4
        x_pos = int(np.cos(angle + phases[i]) * radius + x0)
        y_pos = int(np.sin(angle + phases[i]) * radius + y0)

        brightness = int(128 + 127 * np.sin(phases[i]))
        color = (brightness, brightness, 0)
        ball = pygame.Surface((24, 24), pygame.SRCALPHA, 32).convert_alpha()
        pygame.draw.circle(ball, color, (12, 12), 12)
        ball = add_outer_glow(ball)
        surface.blit(ball, (x_pos - 12, y_pos - 12))


def main() -> None:
    # Окно
    surface = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Kuramoto Fireflies")
    clock = pygame.time.Clock()

    # Модель
    freq = oscillators()
    phases = np.random.uniform(0.0, 2.0 * np.pi, N)

    # UI
    k_slider = Slider(surface, 1375, 75, 50, 150, min=0, max=3, step=0.1, initial=0,
                      handleRadius=12, vertical=True, handleColour=(0, 0, 0))
    output = TextBox(surface, 1362.5, 0, 75, 50, fontSize=30,
                     textColour=(255, 255, 255), colour=(0, 0, 0))

    sync_prev = False
    critical_sync_text = "Critical Point: N/A"

    running = True
    while running:
        k = k_slider.getValue()

        # События
        events = pygame.event.get()
        for e in events:
            if e.type == pygame.QUIT:
                running = False

        # Модель: Эйлер с шагом 0.1
        dt = 0.1
        phases = phases + kuramoto_dynamics(phases, freq, k) * dt
        phases = np.mod(phases, 2.0 * np.pi)  # нормализация фаз

        # Экран
        surface.fill((0, 0, 0))
        plot_scene(phases, surface)

        # Фиксация критической точки
        r = np.abs(np.mean(np.exp(1j * phases)))
        sync = (r + 1.0) / 2.0
        sync_now = sync >= SYNC_THRESHOLD
        if sync_now and not sync_prev:
            critical_sync_text = f"Critical Point: k = {k:.1f}"
        sync_prev = sync_now

        label = FONT.render(critical_sync_text, True, (255, 255, 255))
        surface.blit(label, (SYNC_GRAPH_X + SYNC_GRAPH_X_OFFSET + SYNC_GRAPH_WIDTH + 10, 40))

        output.setText(f"k = {k:.1f}")

        pygame_widgets.update(events)
        pygame.display.update()
        clock.tick(FPS)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
