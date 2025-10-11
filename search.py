"""
search.py

Contains search algorithms for HW1.

Author: Jared Berry
Date: 10/11/2025
"""
import numpy as np
import heapq

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
            if n[0] >= bounds[0][0] and n[0] <= bounds[0][1] and \
               n[1] >= bounds[1][0] and n[1] <= bounds[1][1]:
                neighbors_filtered.append(n)

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
        obstacles: np.ndarray of obstacle locations, rounded to Gridworld resolution
    """
    # Heuristic cost is Euclidean distance
    def h(node):
        return np.linalg.norm(goal - node)
    
    # Initialization
    open = []           # To be evaluated
    node_lookup = {}    # Map positions to nodes
    closed = []         # Already evaluated
    move_cost = 1.0
    obs_cost = 1000.0
    obstacle_set = set(map(tuple, obstacles))
    
    start_node = Node(start, parent=None, obs=False)
    start_node.set_cost(0, h(start))
    node_lookup[tuple(start)] = start_node
    heapq.heappush(open, start_node)

    found = False
    current = start
    while open:
        # Get node in open set with lowest f cost
        current = heapq.heappop(open)
        closed.append(tuple(current.position))

        # Return if current is the goal
        if (current.position == goal).all():
            found = True
            break

        # Iterate over neighbors of current
        neighbors = current.get_neighbor_coords(bounds, res)
        for neigh in neighbors:
            if tuple(neigh) in closed: continue
            
            # Calculate gcost, hcost, and fcost based on current
            obs = tuple(neigh) in obstacle_set
            if obs: neigh_gcost = current.gcost + obs_cost
            else: neigh_gcost = current.gcost + move_cost
            neigh_hcost = h(neigh)

            # If neighbor in set AND new f cost of neighbor is lower OR neighbor not in set
            neigh_node = Node(neigh, parent=current, obs=obs)
            neigh_node.set_cost(neigh_gcost, neigh_hcost)
            if (neigh_node in open and neigh_node < node_lookup[tuple(neigh)]) or\
                neigh_node not in open:
                # Update node lookup and add (or re-add) neighbor to open set
                heapq.heappush(open, neigh_node)
                node_lookup[tuple(neigh)] = neigh_node

    path = []
    if found:
        current = node_lookup[tuple(goal)]
        while current is not None:
            path.append(current.position)
            current = current.parent

        path.reverse()

    return path


