import pygame
import sys
import numpy as np
from numpy.linalg import inv
from control import dlqe

# Initialize pygame
pygame.init()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GT_COLOR = (0, 255, 0)
ESTIMATED_COLOR = (255, 0, 0)
UNFILTRED_COLOR = (0, 0, 255)

# Load the 'Consolas' font
font_size = 20
font = pygame.font.SysFont("firasans", font_size)


# Set up the window
window_width = 800
window_height = 600
window = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Mouse Tracker")


# Set up the clock
clock = pygame.time.Clock()

# Set up variables for mouse position and trail
gt_trail = []  # Ground Truth Trail
estimated_trail = []
unflitred_trail = []

mouse_positionZ1 = np.array([0, 0])


fs = 100  # fps
ts = 1 / fs

A = np.array([[1, 0, ts, 0], [0, 1, 0, ts], [0, 0, 1, 0], [0, 0, 0, 1]])
B = np.zeros((4, 1))
C = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
D = np.zeros((2, 1))


R = 1 * np.eye(4, 4)
Q = 1 * np.array([[1, 0], [0, 1]])

# DEBUG
from control import lqe

G = np.eye(4)
L, P, E = dlqe(A, G, C, R, Q)

from kalman import KF

kf = KF(A, B, C, D, R, Q)


mu = np.array([[0], [0], [0], [0]])
sigma = 10.0 * np.eye(4, 4)

left_click = False
right_click = False

# Main loop
while True:
    window.fill(BLACK)  # Clear the screen

    # Check for events
    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                pass
            if event.button == 3:  # Right click
                pass
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    mouse_buttons = pygame.mouse.get_pressed()
    if mouse_buttons[0]:
        left_click = True
    else:
        left_click = False

    if mouse_buttons[2]:
        right_click = True
    else:
        right_click = False

    delta = np.random.normal(0, 0.1, (2, 1))
    # Get the current mouse position
    mouse_position = np.array(pygame.mouse.get_pos())
    mouse_position_unfiltred = mouse_position.reshape((2, 1)) + delta
    dXdt = (mouse_position - mouse_positionZ1) / ts
    mouse_positionZ1 = mouse_position

    # Add the current mouse position to the trail
    gt_trail.append(mouse_position)
    estimated_trail.append((mu[0][0], mu[1][0]))
    if not left_click:
        unflitred_trail.append((mouse_position_unfiltred[0][0], mouse_position_unfiltred[1][0]))

    # Draw the trail
    for point in gt_trail:
        pygame.draw.circle(window, GT_COLOR, point, 3)

    for point in estimated_trail:
        pygame.draw.circle(window, ESTIMATED_COLOR, point, 3)

    for point in unflitred_trail:
        pygame.draw.circle(window, UNFILTRED_COLOR, point, 3)

    # Limit the length of the trail
    if len(gt_trail) > fs / 100 * 100:
        gt_trail.pop(0)

    if len(estimated_trail) > fs / 100 * 100:
        estimated_trail.pop(0)

    if len(unflitred_trail) > fs / 100 * 100:
        unflitred_trail.pop(0)

    # Draw the current mouse position
    pygame.draw.circle(window, GT_COLOR, mouse_position, 10)

    # Draw estimation
    pygame.draw.circle(window, ESTIMATED_COLOR, (mu[0][0], mu[1][0]), 5)

    # Draw Unflitred position
    if not left_click:
        pygame.draw.circle(window, UNFILTRED_COLOR, (mouse_position_unfiltred[0][0], mouse_position_unfiltred[1][0]), 5)

    # Render the text
    gt_surface = font.render(f"Ground Truth: {mouse_position}", True, GT_COLOR)
    estimated_surface = font.render(f"Estimated: ({mu[0][0]:.2f},{mu[1][0]:.2f})", True, ESTIMATED_COLOR)
    if left_click:
        sensor_info = font.render(f"No sensors !", True, ESTIMATED_COLOR)
    else:
        sensor_info = font.render(f"Sensor readings OK", True, GT_COLOR)

    # Blit the text onto the window
    window.blit(gt_surface, (10, 10))
    window.blit(estimated_surface, (10, 40))
    window.blit(sensor_info, (10, 70))

    # Update the display
    pygame.display.update()

    # Kalman stuff here
    z = np.array([[mouse_position[0]], [mouse_position[1]]])  # Pos
    # z = np.array([[dXdt[0]], [dXdt[1]]]) + delta_t  # Speed
    u = np.array([[0]])
    # mu, sigma = KF_step(mu, sigma, u, z + delta)
    w = [not left_click, not left_click]
    kf.step(u, z + delta, w)
    mu = kf.mu
    sigma = kf.sigma
    # print(kf.sigma)

    # Cap the frame rate
    clock.tick(fs)
