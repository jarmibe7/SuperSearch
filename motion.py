"""
Contains functions for robot path following.

Author: Jared Berry
"""
import numpy as np
from scipy.interpolate import make_splprep, CubicSpline

# Wrap angle to range [-pi, pi]
def wrap_angle(theta):
    return np.arctan2(np.sin(theta), np.cos(theta))

def a_star_to_kspline(a_star_path, degree=3, num_points=500):
    """
    Given an A-start path, compute new interpolated waypoints using a spline of degree k.
    """
    # Compute parameterized spline path
    p = np.array(a_star_path)
    kspline, s = make_splprep([p[:,0], p[:,1]], k=degree)
    s_fine = np.linspace(0, 1, num=num_points, endpoint=True)
    spline_path = kspline(s_fine).T
    return spline_path

def step_rk4(x, u, t, h, w=None):
    """
    A motion model that leverages RK4 integration for improved integration

    Args:
        x: Previous state
        u: Previous control
        t: Current time
        h: Timestep
        w: Noise vector to incorporate into model
    """
    if w is None or w.shape[0] == 0:
        w = np.zeros(x.shape)   # Dummy noise vector

    def f(x, t, u):
        """
        Nonlinear dynamics for planar wheeled robot.

        Args:
            x: State at current timestep
            t: Current timestep
            u_func: Control signal at current timestep
        """
        u_mult = np.array([u[0], u[0], u[1]]) + w
        xdot = np.array([np.cos(x[2]), np.sin(x[2]), 1])
        dyn = xdot * u_mult
        return dyn
    
    k1 = f(x, t, u)
    k2 = f(x + h*k1/2.0, t + h/2.0, u)
    k3 = f(x + h*k2/2.0, t + h/2.0, u)
    k4 = f(x + h*k3, t + h, u)

    return x + h*(k1/6.0 + k2/3.0 + k3/3.0 + k4/6.0)

def sim_rk4(waypoints, kv, kw, h=0.1, interp=True):
    """
    A simulation function that computes a control law at each timestep based on a given
    path.

    Args:
        waypoints: An array of waypoints along a path, computed with A*.
        kv: Gain determining forward velocity magnitude based on position error to desired waypoint.
        kw: Gain determining anglular velocity magnitude based on bearing error to desired waypoint.
        h: Simulation timestep
        interp: Whether to perform spline interpolation on the given path.
    """
    if interp:
        waypoints = a_star_to_kspline(waypoints, degree=3, num_points=500)
    
    # Initialization
    acc_limits = np.array([0.288, 5.5579])
    x0 = np.concatenate([waypoints[0], np.array([-np.pi/2])])
    x_ret = [x0]
    x_curr = x0
    d_thresh = 5e-2     # How close robot must get to a waypoint to move on to next waypoint
    t = 0.0
    v_prev, w_prev = 0.0, 0.0
    for i, x_des in enumerate(waypoints):
        while np.linalg.norm(x_des - x_curr[0:2]) > d_thresh:
            # Compute control signal based on path
            v = kv * np.linalg.norm(x_des - x_curr[0:2])  # Forward velocity based on distance to next waypoint
            w = kw * (np.arctan2(x_des[1] - x_curr[1], x_des[0] - x_curr[0]) - x_curr[2])   # Angular velocity based on bearing to next wayping
            v_lim = np.clip(v, v_prev - acc_limits[0]*h, v_prev + acc_limits[0]*h)
            w_lim = np.clip(w, w_prev - acc_limits[1]*h, w_prev + acc_limits[1]*h)
            u = np.array([v_lim, w_lim])

            # Integrate next timestep
            x_curr = step_rk4(x_curr, u, t, h).copy()
            x_curr[2] = wrap_angle(x_curr[2])
            x_ret.append(x_curr)

            # Update time and previous controls
            t += h
            v_prev = v_lim
            w_prev = w_lim

    return np.array(x_ret)