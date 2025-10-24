"""
search.py

Contains search algorithms for HW1.

Author: Jared Berry
Date: 10/11/2025
"""
import numpy as np
import heapq

from motion import wrap_angle, step_rk4
from utils import inflate_obstacles, round_to_res, pos_to_grid, grid_to_pos

#
# --- A* ---
#
class Node():
    """
    A node in an A* search graph.

    Args:
        pos: The node's position in the graph.
        cost: The node's f-cost
        parent: The node's parent node for the least cost path
        obs: Whether or not the node is an obstacle
    """
    def __init__(self, pos, parent, obs=False):
        self.position = pos
        self.parent = parent
        self.obs = obs

        self.fcost = np.inf
        self.gcost = np.inf
        self.hcost = np.inf

    def get_neighbor_coords(self, bounds, res):
        neighbors = []

        # Cover full immediate neighborhood
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                neigh = (self.position[0] + dx, self.position[1] + dy)
                
                # Skip node coords
                if tuple(neigh) == tuple(self.position):
                    continue

                # Check bounds
                if bounds[0][0] <= neigh[0] < bounds[0][1] and bounds[1][0] <= neigh [1] < bounds[1][1]:
                    neighbors.append(neigh)

        return neighbors


    def set_cost(self, gcost, hcost):
        self.gcost = gcost
        self.hcost = hcost
        self.fcost = self.gcost + self.hcost

    # Comparison and representation
    def __eq__(self, other):    # For checking position equality
        return (self.position == other.position).all()
    def __lt__(self, other):    # For priority queue lookup
        if self.fcost == other.fcost: return self.hcost < other.hcost
        else: return self.fcost < other.fcost
    def __repr__(self):
        return f"Node(position={self.position} fcost={self.fcost} gcost={self.gcost} hcost={self.hcost})"


def a_star(start_rep, 
           goal_rep, 
           bounds_rep, 
           res, 
           obstacles_f, 
           mode='offline',
           obstacles_i=None):
    """
    Use A* path finding to determine the best path from a start to a goal.

    Args:
        start_f: Starting position of robot
        goal_f: Goal position
        bounds_f: Gridworld bounds
        res: Grid resolution
        obstacles_f: Set of obstacle locations, rounded to Gridworld resolution. This
                     param is always needed, for resolving floating point error.
        mode: What mode is it being run in - offline | online | real
        obstacles_i: If online or real, pass inputs in integer form as well 
    """
    # Convert real coords into integer representations
    if mode == 'online': 
        # If running online, integer rep is passed in
        start, goal = start_rep, goal_rep
        obstacles = obstacles_i
        bounds = bounds_rep
    elif mode == 'real':
        start, goal = start_rep, goal_rep
        # start, goal = pos_to_grid(start_rep, res), goal_rep
        obstacles = obstacles_i
        bounds = bounds_rep
    else: 
        start, goal = pos_to_grid(start_rep, res), pos_to_grid(goal_rep, res)
        obstacles = set(pos_to_grid(o, res) for o in obstacles_f)
        bounds = np.array([
            pos_to_grid(bounds_rep[0], res),
            pos_to_grid(bounds_rep[1], res)
        ])

    # Heuristic cost is Euclidean distance
    def h(node):
        return np.linalg.norm(np.array(goal) - np.array(node))
    
    # Initialization
    open_heap = []           # To be evaluated, heapq
    open_set = set()         # Fast lookup to see if nodes are in open set
    node_lookup = {}         # Map positions to nodes
    closed = set()           # Already evaluated
    move_cost = 1.0
    obs_cost = 1000.0
    
    # Add starting node to open set
    start_node = Node(start, parent=None, obs=False)
    start_node.set_cost(0, h(start))
    node_lookup[start] = start_node
    heapq.heappush(open_heap, start_node)
    open_set.add(start)

    found = False
    current = start
    while open_heap:
        # Get node in open set with lowest f cost
        current = heapq.heappop(open_heap)
        curr_pos = current.position
        if curr_pos in closed: continue
        open_set.remove(curr_pos)
        closed.add(curr_pos)

        # Return if current is the goal
        if curr_pos == tuple(goal):
            found = True
            break

        # Iterate over neighbors of current
        neighbors = current.get_neighbor_coords(bounds, res)
        for neigh in neighbors:
            if neigh in closed: continue
            
            # Calculate gcost, hcost, and fcost based on current
            # Check both integer rep. and orig rep. in case of float point error
            obs = (neigh in obstacles) or (tuple(grid_to_pos(neigh, res)) in obstacles_f)
            if obs: neigh_gcost = current.gcost + obs_cost
            else: neigh_gcost = current.gcost + move_cost
            neigh_hcost = h(neigh)

            # If neighbor in set AND new f cost of neighbor is lower OR neighbor not in set
            if neigh not in node_lookup: neigh_node = Node(neigh, parent=current, obs=obs)
            else: neigh_node = node_lookup[neigh]
            if (neigh in open_set and neigh_gcost < node_lookup[neigh].gcost) or\
                neigh not in open_set:
                # Update node lookup and add (or re-add) neighbor to open set
                neigh_node.parent = current
                neigh_node.set_cost(neigh_gcost, neigh_hcost)
                heapq.heappush(open_heap, neigh_node)
                open_set.add(neigh)
                node_lookup[neigh] = neigh_node

    path = []
    if found:
        current = node_lookup[goal]
        while current is not None:
            path.append(tuple(current.position))
            current = current.parent

        path.reverse()

    if mode == 'offline': path = [grid_to_pos(p, res) for p in path]
    return path

def a_star_online(start_f, goal_f, bounds_f, res, obstacles_f):
    """
    Use A* path finding to determine the best path from a start to a goal, planning
    online to avoid obstacles.

    Args:
        start: Starting position of robot
        goal: Goal position
        bounds: Gridworld bounds
        obstacles: np.ndarray of obstacle locations, rounded to Gridworld resolution
    """
    # Initialization
    start, goal = pos_to_grid(start_f, res), pos_to_grid(goal_f, res)
    obstacles = set(pos_to_grid(o, res) for o in obstacles_f)
    bounds = np.array([
        pos_to_grid(bounds_f[0], res),
        pos_to_grid(bounds_f[1], res)
    ])
    path = [tuple(start)]
    known_obstacles_i = set()     # Start without known obstacles
    known_obstacles_f = set()   # Needed to prevent floating point errors

    if tuple(start) in obstacles: return path

    # Continue until goal is reached
    current = start
    while not current == goal:
        naive_path = a_star(current, goal, bounds, res, known_obstacles_f, 
                            mode='online', obstacles_i=known_obstacles_i)
        if not naive_path: break  # No path found

        # Follow naive path
        for node in naive_path[1:]: # First node in path is last node of prev path
            # If node on naive path is obstacle, add to known_obstacles and replan
            if node in obstacles or (tuple(grid_to_pos(node, res)) in obstacles_f):
                known_obstacles_i.add(node)
                known_obstacles_f.add(tuple(grid_to_pos(node, res)))
                # TODO: Check all neighbors of current for obstacles as well
                break
            else:
                # Otherwise, continue on planned path
                current = node
                path.append(current)

    path = [grid_to_pos(p, res) for p in path]
    return path

def a_star_real(start_f, goal_f, bounds_f, res, obstacles_f, kv, kw, h=0.1, noise=0.0, thresh=1e-2, interp=False, obs_avoid=True):
    """
    Use A* path finding to determine the best path from a start to a goal, planning
    online to avoid obstacles.

    Args:
        start: Starting position of robot
        goal: Goal position
        bounds: Gridworld bounds
        obstacles: np.ndarray of obstacle locations, rounded to Gridworld resolution
        kv: Gain determining forward velocity magnitude based on position error to desired waypoint.
        kw: Gain determining anglular velocity magnitude based on bearing error to desired waypoint.
        h: Simulation timestep
        noise: How much noise to add to control signal
        thresh: How close robot must get to a waypoint to set goal to next waypoint in path.
        interp: Whether to interpolate between waypoints
        obs_avoid: Whether to employ extra obstacle avoidance measures
    """
    # A* initialization
    start, goal = pos_to_grid(start_f, res), pos_to_grid(goal_f, res)
    obstacles = set(pos_to_grid(o, res) for o in obstacles_f)
    bounds = np.array([
        pos_to_grid(bounds_f[0], res),
        pos_to_grid(bounds_f[1], res)
    ])
    path = [tuple(start)]
    known_obstacles_i = set()     # Start without known obstacles
    known_obstacles_f = set()   # Needed to prevent floating point errors

    # A* planning and control resolutions are different
    a_res = res
    res = min(res, 0.1)

    # Simulation initialization
    acc_limits = np.array([0.288, 5.5579])
    x0 = np.concatenate([start_f, np.array([-np.pi/2])])
    x_ret = [x0]
    t = 0.0
    v_prev, w_prev = 0.0, 0.0

    if tuple(start) in obstacles: return path

    # Continue until goal is reached
    current = start     # Current node in path
    x_curr = x0         # Current true state
    while not current == goal:
        # Plan a naive path with known obstacls
        naive_path = a_star(current, goal, bounds, a_res, known_obstacles_f, 
                            mode='real', obstacles_i=known_obstacles_i)
        if not naive_path: break  # No path found

        # Check if next node is an obstacle
        for node in naive_path[1:]: # First node in path is last node of prev path
            # If node on naive path is obstacle, add to known_obstacles and replan
            if node in obstacles or (tuple(grid_to_pos(node, a_res)) in obstacles_f):
                known_obstacles_i.add(node)
                known_obstacles_f.add(tuple(grid_to_pos(node, a_res)))
                # TODO: Check all neighbors of current for obstacles as well
                break
            else:
                # Otherwise, continue on planned path
                current = node
                path.append(node)

                # Optionally interoplate between current and starting point
                x_next = grid_to_pos(node, a_res)
                if interp: 
                    x_des_x = np.linspace(x_curr[0], x_next[0], num=30)
                    x_des_y = np.linspace(x_curr[1], x_next[1], num=30)
                    x_des_traj = np.vstack([x_des_x, x_des_y]).T
                else:
                    x_des_traj = [x_next]

                # Inflate obstacles in control resolution grid if using coarse resolution
                if res < a_res and known_obstacles_i:
                    known_obstacles_col = inflate_obstacles(bounds, res, np.array(list(known_obstacles_i)) + a_res/2, inflate=3)
                else:
                    known_obstacles_col = known_obstacles_f

                # Move to next via point
                for x_des in x_des_traj:
                    while np.linalg.norm(x_des - x_curr[0:2]) > thresh:
                        # Compute potential field based on known obstacles
                        if obs_avoid and known_obstacles_col:
                            des_vecs_to_obstacles = np.array([o - x_des for o in known_obstacles_col])
                            des_dist_to_obstacles = np.linalg.norm(des_vecs_to_obstacles, axis=1)
                            des_vecs_to_obstacles /= des_dist_to_obstacles.reshape((des_dist_to_obstacles.shape[0], 1))
                            des_nearby_obs_mask = des_dist_to_obstacles < res
                            for vec, dist in zip(des_vecs_to_obstacles[des_nearby_obs_mask], des_dist_to_obstacles[des_nearby_obs_mask]):
                                x_des -= res*vec
                            
                            # Get vectors to obstacles
                            vecs_to_obstacles = np.array([o - x_curr[0:2] for o in known_obstacles_col])
                            dist_to_obstacles = np.linalg.norm(vecs_to_obstacles, axis=1)
                            vecs_to_obstacles /= dist_to_obstacles.reshape((dist_to_obstacles.shape[0], 1))
                            nearby_obs_mask = dist_to_obstacles < res
                            v_vec = x_des - x_curr[0:2]
                            for vec, dist in zip(vecs_to_obstacles[nearby_obs_mask], dist_to_obstacles[nearby_obs_mask]):
                                v_vec -= 0.05*(1/(dist*res))*vec
                        else:
                            v_vec = x_des - x_curr[0:2]

                        # Compute control signal based on path
                        v = kv * np.linalg.norm(v_vec)  # Forward velocity based on distance to next waypoint
                        w = kw * wrap_angle(np.arctan2(v_vec[1], v_vec[0]) - x_curr[2])   # Angular velocity based on bearing to next wayping
                        v_lim = np.clip(v, v_prev - acc_limits[0]*h, v_prev + acc_limits[0]*h)
                        w_lim = np.clip(w, w_prev - acc_limits[1]*h, w_prev + acc_limits[1]*h)
                        u = np.array([v_lim, w_lim])
                        u += np.random.multivariate_normal(np.zeros(u.shape[0]), noise*np.eye(u.shape[0]))  # Add noise to control signal

                        # Integrate next timestep
                        x_curr = step_rk4(x_curr, u, t, h).copy()
                        x_curr[2] = wrap_angle(x_curr[2])
                        x_ret.append(x_curr)

                        # Update time and previous controls
                        t += h
                        v_prev = v_lim
                        w_prev = w_lim

    path = [grid_to_pos(p, a_res) for p in path]
    return path, np.array(x_ret)