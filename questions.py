"""
questions.py

Contains functions for each question in HW1.

Author: Jared Berry
Date: 10/11/2025
""" 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

from search import round_to_res, a_star, a_star_online, a_star_real
from motion import a_star_to_kspline, sim_rk4

PLOT_PATH = os.path.join(__file__, "..\\figures")
DATA_PATH = os.path.join(__file__, "..\\data")
METRICS_PATH = os.path.join(__file__, "..\\metrics")

#
# --- Plotting ---
#

def plot_search(start, goal, path, bounds, res, obstacles, title, filename, traj=None):
    fig, ax = plot_grid(bounds, res, obstacles, title)

    # Plot path
    for cell in path:
        rect = patches.Rectangle(
            (cell[0], cell[1]), res, res,
            facecolor='red', edgecolor='black'
        )
        ax.add_patch(rect)

    # Plot start and goal
    start_rect = patches.Rectangle(
        (start[0], start[1]), res, res,
        facecolor='blue', edgecolor='black'
    )
    ax.add_patch(start_rect)
    goal_rect = patches.Rectangle(
        (goal[0], goal[1]), res, res,
        facecolor='green', edgecolor='blue'
    )
    ax.add_patch(goal_rect)

    if traj is not None:
        ax.plot(traj[:,0], traj[:,1], color='black', linewidth=4)

    fig_path = os.path.join(PLOT_PATH, filename)
    plt.savefig(fig_path)

def plot_grid(bounds, res, obstacles, title=None):
    # Create figure
    length = int(abs(bounds[0][1] - bounds[0][0]))
    height = int(abs(bounds[1][1] - bounds[1][0]))
    fig, ax = plt.subplots(figsize=(length, height))

    # Set up grid
    x_range = np.arange(bounds[0][0], bounds[0][1] + 1e-9, step=res)    # Add small
    y_range = np.arange(bounds[1][0], bounds[1][1] + 1e-9, step=res)

    # Plot landmarks
    for o in obstacles:
        # Plot obstacles as rectanagles
        rect = patches.Rectangle(
            (o[0], o[1]), res, res,
            facecolor='gray', edgecolor='black'
        )
        ax.add_patch(rect)

    # Set up grid
    ax.set_xticks(x_range)
    ax.set_yticks(y_range)
    ax.grid(color='black', linewidth=0.4)

    # Set up axis labels
    if res >= 0.5:
        ax.set_xticklabels(x_range)
        ax.set_yticklabels(y_range)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_aspect('equal')

    if title is None:
        ax.set_title('A* Gridworld')
        fig_path = os.path.join(PLOT_PATH, 'q1.png')
        plt.savefig(fig_path)
    else:
        ax.set_title(title)

    return fig, ax

def get_obstacles(bounds, res, inflate=0):
    """
    Inflate is the number of cells to inflate in each direction
    """
    # Read ground truth landmark data
    landmarks_truth_data_path = os.path.join(DATA_PATH, 'ds1_Landmark_Groundtruth.dat')
    landmarks_truth = pd.read_csv(landmarks_truth_data_path, sep=r"\s+", comment="#", header=None, names=["subject", "x", "y", "x_sig", "y_sig"])
    landmarks = landmarks_truth.to_numpy()[:, 1:3]
     
    # Plot landmarks
    landmarks_rounded = set()  # Set of landmarks
    # Inflate landmark by inflate number of cells
    for l in landmarks:
        # Cover full square of size (inflate)
        for dx in range(-inflate, inflate + 1):
            for dy in range(-inflate, inflate + 1):
                x, y = (l[0] + dx * res), (l[1] + dy * res)

                l_inf = round_to_res(np.array([x,y]), res)

                # Check bounds
                if bounds[0][0] <= l_inf[0] < bounds[0][1] and bounds[1][0] <= l_inf[1] < bounds[1][1]:
                    landmarks_rounded.add(tuple(l_inf))

    return landmarks_rounded

#
# --- Questions ---
#

def q1():
    print("Running question 1...", end="", flush=True)
    bounds = [
        [-2, 5],    # x bounds
        [-6, 6]     # y bounds
    ]
    res = 1.0
    obstacles = get_obstacles(bounds, res)

    _ = plot_grid(bounds, res, obstacles)
    print("Done\n")

def q2():
    print("Running question 2...", end="", flush=True)
    bounds = [
        [-2, 5],    # x bounds
        [-6, 6]     # y bounds
    ]
    res = 1.0
    obstacles = get_obstacles(bounds, res)

    start = np.array([-2, -6])
    goal = np.array([4, 5])

    path = a_star(start, goal, bounds, res, obstacles)
    plot_search(start, goal, path, bounds, res, obstacles, 'Basic A* Search', 'q2.png')

    print("Done\n")

def q3():
    print("Running question 3...", end="", flush=True)
    bounds = [
        [-2, 5],    # x bounds
        [-6, 6]     # y bounds
    ]
    res = 1.0
    obstacles = get_obstacles(bounds, res)

    start = round_to_res(np.array([0.5, -1.5]), res)
    goal = round_to_res(np.array([0.5, 1.5]), res)
    path = a_star(start, goal, bounds, res, obstacles)
    plot_search(start, goal, path, bounds, res, obstacles, 'Basic A* Search', 'q3a.png')

    start = round_to_res(np.array([4.5, 3.5]), res)
    goal = round_to_res(np.array([4.5, -1.5]), res)
    path = a_star(start, goal, bounds, res, obstacles)
    plot_search(start, goal, path, bounds, res, obstacles, 'Basic A* Search', 'q3b.png')

    start = round_to_res(np.array([-0.5, 5.5]), res)
    goal = round_to_res(np.array([1.5, -3.5]), res)
    path = a_star(start, goal, bounds, res, obstacles)
    plot_search(start, goal, path, bounds, res, obstacles, 'Basic A* Search', 'q3c.png')

    print("Done\n")

def q4():
    print("Running question 4...", end="", flush=True)
    bounds = [
        [-2, 5],    # x bounds
        [-6, 6]     # y bounds
    ]
    res = 1.0
    obstacles = get_obstacles(bounds, res)

    start = np.array([-2, -6])
    goal = np.array([4, 5])
    path = a_star_online(start, goal, bounds, res, obstacles)
    plot_search(start, goal, path, bounds, res, obstacles, 'Online A* Search', 'q4.png')

    print("Done\n")

def q5():
    print("Running question 5...", end="", flush=True)
    bounds = [
        [-2, 5],    # x bounds
        [-6, 6]     # y bounds
    ]
    res = 1.0
    obstacles = get_obstacles(bounds, res)

    start = round_to_res(np.array([0.5, -1.5]), res)
    goal = round_to_res(np.array([0.5, 1.5]), res)
    path = a_star_online(start, goal, bounds, res, obstacles)
    plot_search(start, goal, path, bounds, res, obstacles, 'Basic A* Search', 'q5a.png')

    start = round_to_res(np.array([4.5, 3.5]), res)
    goal = round_to_res(np.array([4.5, -1.5]), res)
    path = a_star_online(start, goal, bounds, res, obstacles)
    plot_search(start, goal, path, bounds, res, obstacles, 'Basic A* Search', 'q5b.png')

    start = round_to_res(np.array([-0.5, 5.5]), res)
    goal = round_to_res(np.array([1.5, -3.5]), res)
    path = a_star_online(start, goal, bounds, res, obstacles)
    plot_search(start, goal, path, bounds, res, obstacles, 'Basic A* Search', 'q5c.png')

    print("Done\n")

def q6():
    print("Running question 6...", end="", flush=True)
    bounds = [
        [-2, 5],    # x bounds
        [-6, 6]     # y bounds
    ]
    res = 0.1

    obstacles = get_obstacles(bounds, res, inflate=3)

    start = np.array([-2, -6])
    goal = np.array([4, 5])
    path = a_star(start, goal, bounds, res, obstacles)
    plot_search(start, goal, path, bounds, res, obstacles, f'Online A* Search - res={res}', 'q6.png')

    print("Done\n")

def q7():
    print("Running question 7...", end="", flush=True)
    bounds = [
        [-2, 5],    # x bounds
        [-6, 6]     # y bounds
    ]
    res = 0.1
    obstacles = get_obstacles(bounds, res, inflate=3)

    start = round_to_res(np.array([2.45, -3.55]), res)
    goal = round_to_res(np.array([0.95, -1.55]), res)
    path = a_star_online(start, goal, bounds, res, obstacles)
    plot_search(start, goal, path, bounds, res, obstacles, f'Online A* Search - res={res}', 'q7a.png')

    start = round_to_res(np.array([4.95, -0.05]), res)
    goal = round_to_res(np.array([2.45, 0.25]), res)
    path = a_star_online(start, goal, bounds, res, obstacles)
    plot_search(start, goal, path, bounds, res, obstacles, f'Online A* Search - res={res}', 'q7b.png')

    start = round_to_res(np.array([-0.55, 1.45]), res)
    goal = round_to_res(np.array([1.95, 3.95]), res)
    path = a_star_online(start, goal, bounds, res, obstacles)
    plot_search(start, goal, path, bounds, res, obstacles, f'Online A* Search - res={res}', 'q7c.png')

    print("Done\n")

def q8():
    print("Running question 8...", end="", flush=True)
    bounds = [
        [-2, 5],    # x bounds
        [-6, 6]     # y bounds
    ]
    res = 0.1

    obstacles = get_obstacles(bounds, res, inflate=3)

    # Compute A* path
    start = np.array([-2, -6])
    goal = np.array([4, 5])
    a_star_path = a_star(start, goal, bounds, res, obstacles)
    x_traj = sim_rk4(a_star_path, kv=1.0, kw=1.0, h=0.1, noise=1e-2, thresh=5e-1)
    plot_search(start, goal, a_star_path, bounds, res, obstacles, f'Interpolated Robot Path with A* Search', 'q8.png', traj=x_traj)

    print("Done\n")

def q9():
    print("Running question 9...", end="", flush=True)
    bounds = [
        [-2, 5],    # x bounds
        [-6, 6]     # y bounds
    ]
    res = 0.1
    obstacles = get_obstacles(bounds, res, inflate=3)

    start = round_to_res(np.array([2.45, -3.55]), res)
    goal = round_to_res(np.array([0.95, -1.55]), res)
    path = a_star_online(start, goal, bounds, res, obstacles)
    x_traj = sim_rk4(path, kv=1.0, kw=1.0, h=0.1, noise=0.00, thresh=5e-2)
    plot_search(start, goal, path, bounds, res, obstacles, f'Interpolated Robot Path with A* Search', 'q9a.png', traj=x_traj)

    start = round_to_res(np.array([4.95, -0.05]), res)
    goal = round_to_res(np.array([2.45, 0.25]), res)
    path = a_star_online(start, goal, bounds, res, obstacles)
    x_traj = sim_rk4(path, kv=1.0, kw=1.0, h=0.1, noise=0.01, thresh=5e-2)
    plot_search(start, goal, path, bounds, res, obstacles, f'Interpolated Robot Path with A* Search', 'q9b.png', traj=x_traj)

    start = round_to_res(np.array([-0.55, 1.45]), res)
    goal = round_to_res(np.array([1.95, 3.95]), res)
    path = a_star_online(start, goal, bounds, res, obstacles)
    x_traj = sim_rk4(path, kv=1.0, kw=1.0, h=0.1, noise=0.01, thresh=5e-2)
    plot_search(start, goal, path, bounds, res, obstacles, f'Interpolated Robot Path with A* Search', 'q9c.png', traj=x_traj)

    print("Done\n")

def q10():
    print("Running question 10...", end="", flush=True)
    bounds = [
        [-2, 5],    # x bounds
        [-6, 6]     # y bounds
    ]
    res = 0.1

    obstacles = get_obstacles(bounds, res, inflate=3)

    # Compute A* path
    start = np.array([-2, -6])
    goal = np.array([4, 5])
    path, x_traj = a_star_real(start, goal, bounds, res, obstacles, kv=1.0, kw=1.0, h=0.01, noise=1e-2, thresh=5e-1)
    plot_search(start, goal, path, bounds, res, obstacles, f'Online Robot Path with A* Search', 'q10.png', traj=x_traj)

    print("Done\n")

def q11():
    print("Running question 11...", end="", flush=True)
    bounds = [
        [-2, 5],    # x bounds
        [-6, 6]     # y bounds
    ]
    res = 0.1
    obstacles = get_obstacles(bounds, res, inflate=3)

    start = round_to_res(np.array([0.5, -1.5]), res)
    goal = round_to_res(np.array([0.5, 1.5]), res)
    path, x_traj = a_star_real(start, goal, bounds, res, obstacles, kv=1.0, kw=1.0, h=0.1, noise=0.00, thresh=9e-2, interp=True)
    plot_search(start, goal, path, bounds, res, obstacles, f'Online Robot Path with A* Search', 'q11a.png', traj=x_traj)

    start = round_to_res(np.array([4.5, 3.5]), res)
    goal = round_to_res(np.array([4.5, -1.5]), res)
    path, x_traj = a_star_real(start, goal, bounds, res, obstacles, kv=1.0, kw=1.0, h=0.1, noise=0.00, thresh=9e-2, interp=True)
    plot_search(start, goal, path, bounds, res, obstacles, f'Online Robot Path with A* Search', 'q11b.png', traj=x_traj)

    start = round_to_res(np.array([-0.5, 5.5]), res)
    goal = round_to_res(np.array([1.5, -3.5]), res)
    path, x_traj = a_star_real(start, goal, bounds, res, obstacles, kv=1.0, kw=1.0, h=0.1, noise=0.00, thresh=9e-2, interp=True)
    plot_search(start, goal, path, bounds, res, obstacles, f'Online Robot Path with A* Search', 'q11c.png', traj=x_traj)

    res = 1.0
    obstacles = get_obstacles(bounds, res, inflate=0)

    start = round_to_res(np.array([0.5, -1.5]), res)
    goal = round_to_res(np.array([0.5, 1.5]), res)
    path, x_traj = a_star_real(start, goal, bounds, res, obstacles, kv=1.0, kw=1.0, h=0.1, noise=0.00, thresh=9e-2, interp=True)
    plot_search(start, goal, path, bounds, res, obstacles, f'Online Robot Path with A* Search', 'q11d.png', traj=x_traj)

    start = round_to_res(np.array([4.5, 3.5]), res)
    goal = round_to_res(np.array([4.5, -1.5]), res)
    path, x_traj = a_star_real(start, goal, bounds, res, obstacles, kv=1.0, kw=1.0, h=0.1, noise=0.00, thresh=9e-2)
    plot_search(start, goal, path, bounds, res, obstacles, f'Online Robot Path with A* Search', 'q11e.png', traj=x_traj)

    start = round_to_res(np.array([-0.5, 5.5]), res)
    goal = round_to_res(np.array([1.5, -3.5]), res)
    path, x_traj = a_star_real(start, goal, bounds, res, obstacles, kv=1.0, kw=1.0, h=0.1, noise=0.00, thresh=9e-2)
    plot_search(start, goal, path, bounds, res, obstacles, f'Online Robot Path with A* Search', 'q11f.png', traj=x_traj)

    print("Done\n")