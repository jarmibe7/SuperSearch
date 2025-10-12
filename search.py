"""
search.py

Contains search algorithms for HW1.

Author: Jared Berry
Date: 10/11/2025
"""
import numpy as np
import heapq

def pos_to_grid(pos, res):
    return tuple(np.floor(np.array(pos) / res).astype(int))

def grid_to_pos(grid, res):
    return np.array(grid) * res

def round_to_res(n, res):
    """
    Given a number or np.ndarray of numbers, round to a given resolution.
    """
    if isinstance(n, tuple): n_arr = np.array(n)
    else: n_arr = n
    return np.round(np.floor(n_arr / res)*res, 1)   # TODO: Better way of eliminating floating point

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

    def round_neighbor_to_res(self, n, res):
        """
        Rounding for neighbors should round to closest resolution interval
        instead of rounding down.
        """
        return np.round(np.round(n / res)*res, 1)

    def get_neighbor_coords(self, bounds, res):
        neighbors = []

        # Cover full immediate neighborhood
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                x, y = (self.position[0] + dx), (self.position[1] + dy)

                # neigh = self.round_neighbor_to_res(np.array([x,y]), res)
                neigh = (x,y)
                
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


def a_star(start_f, goal_f, bounds_f, res, obstacles_f, return_float=True):
    """
    Use A* path finding to determine the best path from a start to a goal.

    Args:
        start_f: Starting position of robot
        goal_f: Goal position
        bounds_f: Gridworld bounds
        res: Grid resolution
        obstacles_f: Set of obstacle locations, rounded to Gridworld resolution
        return_float: Whether to return path in internal int representation or final
                      true units.
    """
    # Convert real coords into integer representations
    start, goal = pos_to_grid(start_f, res), pos_to_grid(goal_f, res)
    obstacles = set(pos_to_grid(o, res) for o in obstacles_f)
    bounds = np.array([
        pos_to_grid(bounds_f[0], res),
        pos_to_grid(bounds_f[1], res)
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
    test_iter = 0   # DEBUG
    while open_heap:
        test_iter += 1  # DEBUG
        # Get node in open set with lowest f cost
        current = heapq.heappop(open_heap)
        # curr_pos = tuple(round_to_res(current.position, res))
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
            # neigh = round_to_res(neigh, res)
            if neigh in closed: continue
            
            # Calculate gcost, hcost, and fcost based on current
            obs = neigh in obstacles
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

    if return_float: path = [grid_to_pos(p, res) for p in path]
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
    start, goal = pos_to_grid(start_f, res), pos_to_grid(goal_f, res)
    obstacles = set(pos_to_grid(o, res) for o in obstacles_f)
    bounds = np.array([
        pos_to_grid(bounds_f[0], res),
        pos_to_grid(bounds_f[1], res)
    ])
    path = [tuple(start)]
    known_obstacles = set() # Start without known obstacles

    if tuple(start) in obstacles: return path

    # Continue until goal is reached
    current = start
    while not current == goal:
        naive_path = a_star(current, goal, bounds, res, known_obstacles, return_float=False)
        if not naive_path: break  # No path found

        # Follow naive path
        for node in naive_path[1:]: # First node in path is last node of prev path
            # If node on naive path is obstacle, add to known_obstacles and replan
            if node in obstacles:
                known_obstacles.add(node)
                # TODO: Check all neighbors of current for obstacles as well
                break
            else:
                # Otherwise, continue on planned path
                current = node
                path.append(current)

    return path

