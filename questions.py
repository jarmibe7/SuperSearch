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

from search import a_star

PLOT_PATH = os.path.join(__file__, "..\\figures")
DATA_PATH = os.path.join(__file__, "..\\data")
METRICS_PATH = os.path.join(__file__, "..\\metrics")

def round_to_res(n, res):
    """
    Given a number or np.ndarray of numbers, round to a given resolution.
    """
    return np.round(n / res)*res

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
    ax.set_xticks(x_range)
    ax.set_yticks(y_range)
    ax.grid(color='black', linewidth=0.4)

    # Set up axis labels
    ax.set_xticklabels(x_range)
    ax.set_yticklabels(y_range)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')

    if title is None:
        ax.set_title('A* Gridworld')
        fig_path = os.path.join(PLOT_PATH, 'q1.png')
        plt.savefig(fig_path)
    else:
        ax.set_title(title)

    return fig, ax

def get_obstacles(res):
    # Read ground truth landmark data
    landmarks_truth_data_path = os.path.join(DATA_PATH, 'ds1_Landmark_Groundtruth.dat')
    landmarks_truth = pd.read_csv(landmarks_truth_data_path, sep=r"\s+", comment="#", header=None, names=["subject", "x", "y", "x_sig", "y_sig"])
    landmarks = landmarks_truth.to_numpy()[:, 1:3]
     
    # Plot landmarks
    landmarks_rounded = np.zeros(landmarks.shape)
    for i, l in enumerate(landmarks):
        l_round = round_to_res(l, res)

        # Take up entire cell and plot rectangle
        if l[0] < l_round[0]: l_round[0] -= res
        if l[1] < l_round[1]: l_round[1] -= res

        landmarks_rounded[i] = l_round

    return landmarks_rounded

def build_gridworld(bounds, res=1.0):
    x = np.arange(bounds[0][0], bounds[0][1], step=res)
    y = np.arange(bounds[1][0], bounds[1][1], step=res)
    X, Y = np.meshgrid(x, y)

    return X, Y

def q1():
    print("Running question 1...", end="", flush=True)
    bounds = [
        [-2, 5],    # x bounds
        [-6, 6]     # y bounds
    ]
    res = 1.0
    obstacles = get_obstacles(res)

    _ = plot_grid(bounds, res, obstacles)
    print("Done\n")

def q2():
    print("Running question 2...", end="", flush=True)
    bounds = [
        [-2, -6],    # x bounds
        [5,   6]     # y bounds
    ]
    res = 1.0
    obstacles = get_obstacles(res)

    start = np.array([-2, 6])
    goal = np.array([5, 6])

    path = a_star(start, goal, bounds, res, obstacles)
    plot_search(start, goal, path, bounds, res, obstacles, 'Basic A* Search', 'q2.png')

    print("Done\n")
