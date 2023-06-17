import sys
import pygame
import numpy as np
import pygame_widgets
from pygame_widgets.slider import Slider
from pygame_widgets.textbox import TextBox
from pygame.font import SysFont
import matplotlib.pyplot as plt
N = 15
w0 = 0.5
sigma = 0.3
width = 1500
height = 800

# Функция, которая возвращает частоту мигания для каждого светлячка
def oscillators():
    return w0 + sigma * np.random.uniform(-1, 1, N)

sync_values = []
sync_graph_x = 10
sync_graph_y = 50
sync_graph_scale = 200
sync_graph_step = 0.1
sync_graph_width = 250
sync_graph_x_offset = 20
tick_font_size = 12
lit_fireflies_values = []
common_graph_x = 10
common_graph_y = 50
common_graph_scale = 200
common_graph_step = 0.1
common_graph_width = 250
common_graph_x_offset = 20
sync_threshold = 0.99     
sync_time_sec = 5          
sync_time_frames = int(sync_time_sec * 60)  
sync_check_counter = 0     

sync_prev_status = False  
critical_sync = "Critical Point: N/A"
pygame.init()

time = 0
font = pygame.font.Font(None, 36)
tick_font = SysFont('Arial', tick_font_size)



def add_outer_glow(image, color=(0, 191, 255), size=4):
    new_image = image.copy()
    for _ in range(size):
        temp_image = new_image.copy()
        temp_image.fill(0)
        pygame.draw.circle(temp_image, color, (12, 12,), 12-_)
        new_image.blit(temp_image, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)
    return new_image



def interpolate_color(color1, color2, factor):
    return (
        int(color1[0] * (1 - factor) + color2[0] * factor),
        int(color1[1] * (1 - factor) + color2[1] * factor),
        int(color1[2] * (1 - factor) + color2[2] * factor),
    )


def kuramoto_dynamics(phases, freq, k, N):
    dxdt = np.zeros(N)
    for i in range(N):
        sin_sum = np.sum(np.sin(phases - phases[i]))
        dxdt[i] = freq[i] + k / N * sin_sum
    return dxdt



def plot_sync(phases, surface, graph_y):
    sync = (np.abs(np.mean(np.exp(1j * phases))) + 1) / 2
    sync_values.append(sync)
    
    if len(sync_values) > sync_graph_width:
        sync_values.pop(0)
    
    sync_label = font.render("Sync: {:.3f}".format(sync), True, (255, 255, 255))
    surface.blit(sync_label, (10, 10))

    # Рисование графика изменения параметра Sync
    for i in range(1, len(sync_values)):
        color_factor = sync_values[i] / max(sync_values)
        color = interpolate_color((255, 0, 0), (0, 0, 255), color_factor)
        pygame.draw.line(
            surface, color,
            (sync_graph_x + sync_graph_x_offset + i - 1, sync_graph_y + sync_graph_scale - sync_values[i - 1] * sync_graph_scale),
            (sync_graph_x + sync_graph_x_offset + i, sync_graph_y + sync_graph_scale - sync_values[i] * sync_graph_scale),
            3
        )


def plot_lit_fireflies(phases, surface, graph_y):

    lit_fireflies = sum(np.sin(phases) > 0)
    lit_fireflies_values.append(lit_fireflies)

    if len(lit_fireflies_values) > common_graph_width:
        lit_fireflies_values.pop(0)

    lit_fireflies_label = font.render("Lit fireflies: {:d}".format(lit_fireflies), True, (255, 255, 255))
    surface.blit(lit_fireflies_label, (10,  graph_y - 45))

    for i in range(1, len(lit_fireflies_values)):
        pygame.draw.line(
            surface, (0, 255, 0),
            (common_graph_x + common_graph_x_offset + i - 1, graph_y + common_graph_scale - lit_fireflies_values[i - 1] * common_graph_scale / N),
            (common_graph_x + common_graph_x_offset + i, graph_y + common_graph_scale - lit_fireflies_values[i] * common_graph_scale / N),
            1
        )



def plot_oscillators(phases, time, surface):

    plot_sync(phases, surface,sync_graph_y )
    
    plot_lit_fireflies(phases, surface, common_graph_y + common_graph_scale + 250)

    pygame.draw.line(surface, (255, 255, 255), (20, 500), 
    (20,720), 1)

    pygame.draw.line(surface, (255, 255, 255), (20, 720), 
    (280,720), 1)


    # Рисование осей координат и меток деления
    pygame.draw.line(surface, (255, 255, 255), (sync_graph_x + sync_graph_x_offset, sync_graph_y), 
    (sync_graph_x + sync_graph_x_offset, sync_graph_y + sync_graph_scale), 1)
    pygame.draw.line(surface, (255, 255, 255), (sync_graph_x + sync_graph_x_offset, sync_graph_y + sync_graph_scale), 
    (sync_graph_x + sync_graph_x_offset + sync_graph_width, sync_graph_y + sync_graph_scale), 1)
    
    y_ticks = int(sync_graph_scale / (sync_graph_step * sync_graph_scale))
    for i in range(1, y_ticks):
        pygame.draw.line(surface, (255, 255, 255), 
        (sync_graph_x + sync_graph_x_offset - 5, sync_graph_y + sync_graph_scale - i * sync_graph_step * sync_graph_scale), 
        (sync_graph_x + sync_graph_x_offset, sync_graph_y + sync_graph_scale - i * sync_graph_step * sync_graph_scale), 1)
        tick_label = tick_font.render("{:.1f}".format(i * sync_graph_step), True, (255, 255, 255))
        surface.blit(tick_label, (sync_graph_x + sync_graph_x_offset - 30, sync_graph_y + sync_graph_scale - i * sync_graph_step * sync_graph_scale - 10))

    
    for i in range(N):
        x0 = width/2 + 100*np.cos(i / N * 2*np.pi)
        y0 = height/2 + 100*np.sin(i / N * 2*np.pi)
        
        radius = 100
        angle = (i / N * 2*np.pi) + np.pi/4
        x_pos = int(np.cos(angle + phases[i]) * radius + x0)
        y_pos = int(np.sin(angle + phases[i]) * radius + y0)

        brightness = int(128 + 127 * np.sin(phases[i]))
        color = (brightness, brightness, 0)
        ball_surface = pygame.Surface((24, 24), pygame.SRCALPHA, 32).convert_alpha()

        pygame.draw.circle(ball_surface, color, (12, 12), 12)
        ball_surface_with_outer_glow = add_outer_glow(ball_surface)
        surface.blit(ball_surface_with_outer_glow, (x_pos-12, y_pos-12))



pygame.init()
surface = pygame.display.set_mode((width, height))
time = 0
font = pygame.font.Font(None, 36)

freq = oscillators()
phases = np.random.uniform(0, 2 * np.pi, N)

clock = pygame.time.Clock()

k_slider = Slider(surface, 1375, 75, 50, 150, min=0, max=3, step=0.1, initial=0, handleRadius = 12, vertical=True, handleColour = (0,0,0))
output = TextBox(surface, 1362.5, 0, 75, 50, fontSize=30, textColour = (255,255,255), colour = (0,0,0))

while True:
    k = k_slider.getValue()
    clock.tick(60)
    phases += kuramoto_dynamics(phases, freq, k, N) * 0.1
    time += 0.01
    pygame.time.delay(10)

    sync = (np.abs(np.mean(np.exp(1j * phases))) + 1) / 2
    
    sync_status = sync >= sync_threshold
    if sync_status and not sync_prev_status:
        critical_sync = f"Critical Point: k = {k:.1f}"
    sync_prev_status = sync_status
    
    critical_sync_label = font.render(critical_sync, True, (255, 255, 255))
    
    output.setText("k = " + str(round(k_slider.getValue(),1)))
    surface.fill((0,0,0))
    plot_oscillators(phases, time, surface)
    surface.blit(critical_sync_label, (sync_graph_x + sync_graph_x_offset + sync_graph_width + 10, 40))

    events = pygame.event.get()
    
    for event in events:
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
    pygame_widgets.update(events)
    pygame.display.update()