import json
import numpy as np
from matplotlib.patches import Circle
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

from constants import REACH_ZONE_R, MAP_W, MAP_H, NODE_R, GOAL_R, \
                      AGENT_R, FOV_R


class Env():

    def __init__(self):
        self.map_id = None
        self.map = None
        self.kdtree = None
        self.tree_ids = None  # Array of IDs matching kdtree.data
        self.node_lookup = {}  # Node ID -> node obj (both nodes/goals)
        self.goal_ids = []
        self.origin_node_id = None

    def load_map(self, id=1):
        self.map_id = id
        with open('../maps/%d.json' % id, 'r') as f:
            self.map = json.load(f)
            points = []
            ids = []
            for n in self.map.get("nodes"):
                point = [n.get('x'), n.get('y')]
                points.append(point)
                id = n.get('id')
                self.node_lookup[id] = n
                ids.append(id)
                if n.get('type') == 'goal':
                    self.goal_ids.append(id)
                if point == [0, 0]:
                    self.origin_node_id = id
            points = np.array(points)
            self.kdtree = cKDTree(points)
            self.tree_ids = np.array(ids)

    def goals(self):
        return [self.node(id) for id in self.goal_ids]

    def node(self, id):
        return self.node_lookup.get(id)

    def node_loc(self, id):
        n = self.node(id)
        if n:
            return np.array([n.get('x'), n.get('y')])

    def nearest_points(self, loc, r=FOV_R, return_sorted=False):
        idxs = self.kdtree.query_ball_point(loc, r, return_sorted=return_sorted)
        points = self.kdtree.data[idxs, :]
        return points, self.tree_ids[idxs]

    def nearest_node(self, loc):
        dist, idx = self.kdtree.query(loc, k=1)
        id = self.tree_ids[idx]
        return self.node(id), dist

    def nearest_goal(self, loc, r=3*FOV_R):
        points, ids = self.nearest_points(loc, r=r, return_sorted=True)
        for id in ids:
            if id in self.goal_ids:
                return self.node(id)

    def is_near_goal(self, loc):
        g = self.nearest_goal(loc, r=GOAL_R * 1.5)
        return g is not None

    def is_within_node(self, loc, lenience=1.0):
        """
        Returns nodes (goal or node) if loc is within a tolerance of closest node
        """
        n, dist = self.nearest_node(loc)
        is_goal = n.get('type') == 'goal'
        r = GOAL_R if is_goal else NODE_R
        if dist <= (1+lenience) * r:
            return n

    def observation(self, loc):
        """
        Return nodes in observable foveal radius in relative coordinates.
        Sorting?
        """
        points, ids = self.nearest_points(loc, r=FOV_R + NODE_R)
        return [self.node(id) for id in ids]

    def render_map(self, render_nodes=True, agent_loc=None, fovea_loc=None, path=None, ax=None):
        m = self.map
        if not ax:
            fig, ax = plt.subplots(dpi=144, figsize=(MAP_W/60, MAP_H/60))
        nodes = m.get("nodes")
        for n in nodes:
            is_goal = n.get('type') == 'goal'
            if render_nodes or is_goal:
                x, y = n.get('x'), n.get('y')
                s = GOAL_R if is_goal else NODE_R
                ax.add_patch(Circle((x, y), s,
                                    fill=False, lw=1, alpha=0.5,
                                    edgecolor='green' if is_goal else 'black'))
        if fovea_loc is not None:
            ax.add_patch(Circle(fovea_loc, FOV_R, fill=False, edgecolor='yellow'))
        if agent_loc is not None:
            ax.add_patch(Circle(agent_loc, AGENT_R, fill=True, facecolor='blue', alpha=0.5))
            reach_zone = Circle(agent_loc, REACH_ZONE_R, fill=False, color='black', alpha=0.2)
            ax.add_patch(reach_zone)
        if path:
            X, Y = [], []
            for loc in path:
                X.append(loc[0])
                Y.append(loc[1])
            ax.plot(X, Y, lw=2, color='black', dashes=[2, 2])
        ax.set_xlim((-MAP_W/2, MAP_W/2))
        ax.set_ylim((-MAP_H/2, MAP_H/2))
        return ax

