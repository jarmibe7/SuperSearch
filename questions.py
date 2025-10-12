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

from search import a_star, a_star_online

PLOT_PATH = os.path.join(__file__, "..\\figures")
DATA_PATH = os.path.join(__file__, "..\\data")
METRICS_PATH = os.path.join(__file__, "..\\metrics")

def round_to_res(n, res):
    """
    Given a number or np.ndarray of numbers, round to a given resolution.
    """
    return np.floor(n / res)*res

def plot_search(start, goal, path, bounds, res, obstacles, title, filename):
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
        facecolor='green', edgecolor='black'
    )
    ax.add_patch(goal_rect)

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

    # Set up axis labels
    if res >= 0.5:
        ax.set_xticklabels(x_range)
        ax.set_yticklabels(y_range)
    ax.set_xticks(x_range)
    ax.set_yticks(y_range)
    ax.grid(color='black', linewidth=0.4)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')

    if title is None:
        ax.set_title('A* Gridworld')
        fig_path = os.path.join(PLOT_PATH, 'q1.png')
        plt.savefig(fig_path)
    else:
        ax.set_title(title)

    return fig, ax

def get_obstacles(bounds, res, inflate=1.0):
    # Read ground truth landmark data
    landmarks_truth_data_path = os.path.join(DATA_PATH, 'ds1_Landmark_Groundtruth.dat')
    landmarks_truth = pd.read_csv(landmarks_truth_data_path, sep=r"\s+", comment="#", header=None, names=["subject", "x", "y", "x_sig", "y_sig"])
    landmarks = landmarks_truth.to_numpy()[:, 1:3]
     
    # Plot landmarks
    landmarks_rounded = set()  # Set of landmarks
    for i, l in enumerate(landmarks):
        l_round = round_to_res(l, res)

        l_center = l_round + res/2
        r = (np.sqrt(inflate)/2)*res - 1e-8
        inflate_l_bounds = round_to_res(np.array([
            l_center - r,
            l_center + r 
        ]), res)

        # Check all potential cells covered by inflated size
        for xi in np.arange(inflate_l_bounds[0][0], inflate_l_bounds[1][0]+1, step=res):
            for yi in np.arange(inflate_l_bounds[0][1], inflate_l_bounds[1][1]+1, step=res):
                new_l_node = (xi, yi)
                # Check x bounds, y bounds, and duplicate
                if xi >= bounds[0][0] and xi < bounds[0][1] and \
                yi >= bounds[1][0] and yi < bounds[1][1] and \
                new_l_node not in landmarks_rounded:
                
                    landmarks_rounded.add(tuple(new_l_node))

    return landmarks_rounded

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
    res = 1.0
    obstacles = get_obstacles(bounds, res, inflate=1.0)


    start = np.array([-2, -6])
    goal = np.array([4, 5])
    path = a_star(start, goal, bounds, res, obstacles)
    plot_search(start, goal, path, bounds, res, obstacles, f'Online A* Search - res={res}', 'q6.png')

    print("Done\n")
