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
import matplotlib.path as mpl_path
import os
import json
import time

from search import a_star, a_star_online, a_star_real
from motion import a_star_to_kspline, sim_rk4
from utils import get_obstacles, round_to_res

PLOT_PATH = os.path.join(__file__, "..\\figures")
DATA_PATH = os.path.join(__file__, "..\\data")
METRICS_PATH = os.path.join(__file__, "..\\metrics")

# 
# --- Evaluation ---
#
#
# --- Evaluation Metrics ---
#
def t_match(traj, num_samples):
    """
    Resample a trajectory to have a certain number of samples
    """
    old_path_idx = np.linspace(0, 1, traj.shape[0])
    new_path_idx = np.linspace(0, 1, num_samples)

    traj_resamp = np.column_stack([
        np.interp(new_path_idx, old_path_idx, traj[:, i]) for i in range(traj.shape[1])
    ])

    return traj_resamp

def rmse(predicted, actual, angle=False):
    """
    Given two 1D numpy arrays of the same length, compute Root Mean Squared Error
    between them.
    """
    if angle: error = np.unwrap(actual - predicted)
    else: error = np.linalg.norm(actual - predicted)
    return np.sqrt(error)

def compute_traj_statistics(predicted, actual):
    """
    Given a trajectory, compute various statistics about it from a ground truth.
    """
    stats = {}
    stats['rmse_x'] = rmse(predicted[:, 0], actual[:, 0])
    stats['rmse_y'] = rmse(predicted[:, 1], actual[:, 1])
    stats['rmse_theta'] = rmse(predicted[:, 2], actual[:, 2])
    stats['corr_x'] = np.corrcoef(predicted[:, 0], actual[:, 0])[0, 1]
    stats['corr_y'] = np.corrcoef(predicted[:, 1], actual[:, 1])[0, 1]
    stats['corr_theta'] = np.corrcoef(predicted[:, 2], actual[:, 2])[0, 1]

    return stats

#
# --- Plotting ---
#
def plot_search(start, goal, path, bounds, res, obstacles, title, filename, traj=None, display_robot=True):
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
        facecolor='purple', edgecolor='black'
    )
    ax.add_patch(start_rect)
    goal_rect = patches.Rectangle(
        (goal[0], goal[1]), res, res,
        facecolor='green', edgecolor='black'
    )
    ax.add_patch(goal_rect)

    if traj is not None:
        ax.plot(traj[:,0], traj[:,1], color='black', linewidth=4)
    
    if traj is not None and display_robot:
        arrow = np.array([[0.1, 0.3], [0.1, -0.3], 
                          [1.0, 0.0], [0.1, 0.3]])  # arrow shape
        color = 'dodgerblue'
        for i, xt in enumerate(traj):
            if i % (traj.shape[0] // 15) == 0 or i == 0:
                # Plot robot location and heading
                R = np.array([[np.cos(xt[2]), np.sin(xt[2])],
                                            [-np.sin(xt[2]), np.cos(xt[2])]])
                arrow_rot = 2*arrow @ R
                codes = [mpl_path.Path.MOVETO, mpl_path.Path.LINETO, mpl_path.Path.LINETO, mpl_path.Path.CLOSEPOLY]
                arrow_head_marker = mpl_path.Path(arrow_rot, codes)
                ax.plot(xt[0], xt[1], linestyle='', marker=arrow_head_marker, markersize=20, color=color)

    fig_path = os.path.join(PLOT_PATH, filename)
    plt.savefig(fig_path)
    plt.close()

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
    
    # Set axis limits
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])

    if title is None:
        ax.set_title('A* Gridworld')
        fig_path = os.path.join(PLOT_PATH, 'q1.png')
        plt.savefig(fig_path)
    else:
        ax.set_title(title)

    return fig, ax

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
    x_traj = sim_rk4(a_star_path, kv=1.0, kw=1.0, h=0.1, noise=1e-2, thresh=9e-2)
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
    path, x_traj = a_star_real(start, goal, bounds, res, obstacles, kv=1.0, kw=1.0, h=0.01, noise=1e-2, thresh=0.1)
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

    noi = 1e-2

    start = round_to_res(np.array([0.5, -1.5]), res)
    goal = round_to_res(np.array([0.5, 1.5]), res)
    path, x_traj = a_star_real(start, goal, bounds, res, obstacles, kv=1.0, kw=1.0, h=0.1, noise=noi, thresh=9e-2, interp=True)
    plot_search(start, goal, path, bounds, res, obstacles, f'Online Robot Path with A* Search', 'q11a.png', traj=x_traj, display_robot=True)

    start = round_to_res(np.array([4.5, 3.5]), res)
    goal = round_to_res(np.array([4.5, -1.5]), res)
    path, x_traj = a_star_real(start, goal, bounds, res, obstacles, kv=1.0, kw=1.0, h=0.1, noise=noi, thresh=9e-2, interp=True)
    plot_search(start, goal, path, bounds, res, obstacles, f'Online Robot Path with A* Search', 'q11b.png', traj=x_traj, display_robot=True)

    start = round_to_res(np.array([-0.5, 5.5]), res)
    goal = round_to_res(np.array([1.5, -3.5]), res)
    path, x_traj = a_star_real(start, goal, bounds, res, obstacles, kv=1.0, kw=1.0, h=0.1, noise=noi, thresh=9e-2, interp=True)
    plot_search(start, goal, path, bounds, res, obstacles, f'Online Robot Path with A* Search', 'q11c.png', traj=x_traj, display_robot=True)

    res = 1.0
    obstacles = get_obstacles(bounds, res, inflate=0)

    start = round_to_res(np.array([0.5, -1.5]), res)
    goal = round_to_res(np.array([0.5, 1.5]), res)
    path, x_traj = a_star_real(start, goal, bounds, res, obstacles, kv=1.0, kw=1.0, h=0.1, noise=noi, thresh=9e-2, interp=True)
    plot_search(start, goal, path, bounds, res, obstacles, f'Online Robot Path with A* Search', 'q11d.png', traj=x_traj)

    start = round_to_res(np.array([4.5, 3.5]), res)
    goal = round_to_res(np.array([4.5, -1.5]), res)
    path, x_traj = a_star_real(start, goal, bounds, res, obstacles, kv=1.0, kw=1.0, h=0.1, noise=noi, thresh=9e-2, interp=True)
    plot_search(start, goal, path, bounds, res, obstacles, f'Online Robot Path with A* Search', 'q11e.png', traj=x_traj)

    start = round_to_res(np.array([-0.5, 5.5]), res)
    goal = round_to_res(np.array([1.5, -3.5]), res)
    path, x_traj = a_star_real(start, goal, bounds, res, obstacles, kv=1.0, kw=1.0, h=0.1, noise=noi, thresh=9e-2, interp=True)
    plot_search(start, goal, path, bounds, res, obstacles, f'Online Robot Path with A* Search', 'q11f.png', traj=x_traj)

    print("Done\n")

def noise():
    print("Running noise comparison...", end="", flush=True)
    bounds = [
        [-2, 5],    # x bounds
        [-6, 6]     # y bounds
    ]
    res = 0.1
    obstacles = get_obstacles(bounds, res, inflate=3)


    start = round_to_res(np.array([-0.5, 5.5]), res)
    goal = round_to_res(np.array([1.5, -3.5]), res)
    path, x_traj_gt = a_star_real(start, goal, bounds, res, obstacles, kv=1.0, kw=1.0, h=0.1, noise=0.0, thresh=9e-2, interp=True)
    plot_search(start, goal, path, bounds, res, obstacles, f'Online Robot Path with A* Search - Low Noise', 'noise_none.png', traj=x_traj_gt, display_robot=True)

    path, x_traj_med = a_star_real(start, goal, bounds, res, obstacles, kv=1.0, kw=1.0, h=0.1, noise=0.01, thresh=9e-2, interp=True)
    plot_search(start, goal, path, bounds, res, obstacles, f'Online Robot Path with A* Search - Moderate Noise', 'noise_med.png', traj=x_traj_med, display_robot=True)

    path, x_traj_high = a_star_real(start, goal, bounds, res, obstacles, kv=1.0, kw=1.0, h=0.1, noise=0.1, thresh=9e-2, interp=True)
    plot_search(start, goal, path, bounds, res, obstacles, f'Online Robot Path with A* Search - High Noise', 'noise_high.png', traj=x_traj_high, display_robot=True)

    # Compute statistics
    num_samples = x_traj_gt.shape[0]
    traj_med_resamp = t_match(x_traj_med, num_samples)
    traj_high_resamp = t_match(x_traj_high, num_samples)
    traj_gt_resamp = t_match(x_traj_gt, num_samples)
    stats_med = compute_traj_statistics(traj_med_resamp, traj_gt_resamp)
    stats_high = compute_traj_statistics(traj_high_resamp, traj_gt_resamp)

    metrics_dict = {'med': stats_med, 'high': stats_high}
    for key, value in metrics_dict.items():
        met_path = os.path.join(METRICS_PATH, f'{key}_metrics.json')
        with open(met_path, "w") as f:
            json.dump(value, f, indent=4)

    print("Done\n")

def res_comp():
    print("Running resolution comparison...", end="", flush=True)
    bounds = [
        [-2, 5],    # x bounds
        [-6, 6]     # y bounds
    ]
    res = 0.1
    obstacles = get_obstacles(bounds, res, inflate=3)
    noi = 0.0

    start = round_to_res(np.array([-0.5, 5.5]), res)
    goal = round_to_res(np.array([1.5, -3.5]), res)
    path = a_star(start, goal, bounds, res, obstacles)
    x_traj_gt = sim_rk4(path, kv=1.0, kw=1.0, h=0.1, noise=noi, thresh=9e-2)
    plot_search(start, goal, path, bounds, res, obstacles, f'Offline Fine Resolution', 'off_fine.png', traj=x_traj_gt, display_robot=True)

    start_time = time.time()
    path, x_traj_fine = a_star_real(start, goal, bounds, res, obstacles, kv=1.0, kw=1.0, h=0.1, noise=noi, thresh=9e-2, interp=True)
    end_time = time.time()
    fine_duration = end_time - start_time
    plot_search(start, goal, path, bounds, res, obstacles, f'Online Fine Resolution', 'on_fine.png', traj=x_traj_fine, display_robot=True)

    res = 1.0
    obstacles = get_obstacles(bounds, res, inflate=0)
    start = round_to_res(np.array([-0.5, 5.5]), res)
    goal = round_to_res(np.array([1.5, -3.5]), res)

    start_time = time.time()
    path, x_traj_coarse = a_star_real(start, goal, bounds, res, obstacles, kv=1.0, kw=1.0, h=0.1, noise=noi, thresh=9e-2, interp=True)
    end_time = time.time()
    coarse_duration = end_time - start_time
    plot_search(start, goal, path, bounds, res, obstacles, f'Online Coarse Resolution', 'on_coarse.png', traj=x_traj_coarse, display_robot=True)

    # Compute statistics
    num_samples = x_traj_gt.shape[0]
    traj_fine_resamp = t_match(x_traj_fine, num_samples)
    traj_coarse_resamp = t_match(x_traj_coarse, num_samples)
    traj_gt_resamp = t_match(x_traj_gt, num_samples)
    stats_fine = compute_traj_statistics(traj_fine_resamp, traj_gt_resamp)
    stats_coarse = compute_traj_statistics(traj_coarse_resamp, traj_gt_resamp)
    stats_fine['runtime'] = fine_duration
    stats_coarse['runtime'] = coarse_duration

    metrics_dict = {'fine': stats_fine, 'coarse': stats_coarse}
    for key, value in metrics_dict.items():
        met_path = os.path.join(METRICS_PATH, f'{key}_metrics.json')
        with open(met_path, "w") as f:
            json.dump(value, f, indent=4)
    print("Done\n")