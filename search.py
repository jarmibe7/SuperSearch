"""
search.py

Contains search algorithms for HW1.

Author: Jared Berry
Date: 10/11/2025
"""
import numpy as np
import heapq

def round_to_res(n, res):
    """
    Given a number or np.ndarray of numbers, round to a given resolution.
    """
    idx = np.round(n / res).astype(int)
    # Convert back to actual value
    return idx * res

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
        neighbors.append(self.position + np.array([0.0, res]))    # Top
        neighbors.append(self.position + np.array([res, res]))    # Top right
        neighbors.append(self.position + np.array([res, 0.0]))    # Right
        neighbors.append(self.position + np.array([res, -res]))   # Bottom right
        neighbors.append(self.position + np.array([0.0, -res]))   # Bottom
        neighbors.append(self.position + np.array([-res, -res]))  # Bottom left
        neighbors.append(self.position + np.array([-res, 0.0]))   # Left
        neighbors.append(self.position + np.array([-res, res]))   # Top left
        
        # Check bounds
        neighbors_filtered = []
        for n in neighbors:
            if n[0] >= bounds[0][0] and n[0] < bounds[0][1] and \
               n[1] >= bounds[1][0] and n[1] < bounds[1][1]:
                neighbors_filtered.append(round_to_res(n, res))

        return neighbors_filtered


    def set_cost(self, gcost, hcost):
        if not self.obs:
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


def a_star(start, goal, bounds, res, obstacles):
    """
    Use A* path finding to determine the best path from a start to a goal.

    Args:
        start: Starting position of robot
        goal: Goal position
        bounds: Gridworld bounds
        obstacles: Set of obstacle locations, rounded to Gridworld resolution
    """
    # Round to given resolution
    start, goal = round_to_res(start, res), round_to_res(goal, res)

    # Heuristic cost is Euclidean distance
    def h(node):
        return np.linalg.norm(goal - node)
    
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
    node_lookup[tuple(start)] = start_node
    heapq.heappush(open_heap, start_node)
    open_set.add(tuple(start))

    found = False
    current = start
    test_iter = 0
    while open_heap:
        test_iter += 1
        # Get node in open set with lowest f cost
        current = heapq.heappop(open_heap)
        curr_pos = tuple(round_to_res(current.position, res))
        open_set.remove(curr_pos)
        closed.add(curr_pos)

        # Return if current is the goal
        if curr_pos == tuple(goal):
            found = True
            break

        # Iterate over neighbors of current
        neighbors = current.get_neighbor_coords(bounds, res)
        for neigh in neighbors:
            neigh = round_to_res(neigh, res)
            if tuple(neigh) in closed: continue
            
            # Calculate gcost, hcost, and fcost based on current
            obs = tuple(neigh) in obstacles
            if obs: neigh_gcost = current.gcost + obs_cost
            else: neigh_gcost = current.gcost + move_cost
            neigh_hcost = h(neigh)

            # If neighbor in set AND new f cost of neighbor is lower OR neighbor not in set
            if tuple(neigh) not in node_lookup: neigh_node = Node(neigh, parent=current, obs=obs)
            else: neigh_node = node_lookup[tuple(neigh)]
            if (tuple(neigh) in open_set and neigh_gcost < node_lookup[tuple(neigh)].gcost) or\
                tuple(neigh) not in open_set:
                # Update node lookup and add (or re-add) neighbor to open set
                neigh_node.parent = current
                neigh_node.set_cost(neigh_gcost, neigh_hcost)
                heapq.heappush(open_heap, neigh_node)
                open_set.add(tuple(neigh))
                node_lookup[tuple(neigh)] = neigh_node

    path = []
    if found:
        current = node_lookup[tuple(goal)]
        while current is not None:
            path.append(tuple(current.position))
            current = current.parent

        path.reverse()

    return path

def a_star_online(start, goal, bounds, res, obstacles):
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
    path = [tuple(start)]
    known_obstacles = set() # Start without known obstacles

    if tuple(start) in obstacles: return path

    # Continue until goal is reached
    current = start
    while not (current == goal).all():
        naive_path = a_star(current, goal, bounds, res, known_obstacles)
        if not naive_path: break  # No path found

        # Follow naive path
        for node in naive_path:
            # If node on naive path is obstacle, add to known_obstacles and replan
            if node in obstacles:
                known_obstacles.add(node)
                # TODO: Check all neighbors of current for obstacles as well
                break
            elif node not in path:
                # Otherwise, continue on planned path
                current = node
                path.append(current)

    return path

