import numpy as np

from constants import REACH_ZONE_R, REACHABLE_BUFFER, MAP_W


class Agent():
    def __init__(self, runner=None, env=None, verbose=False, **kwargs):
        self.env = env
        self.runner = runner  # TaskRunner (optional)

        # State
        self.loc = np.array([0.0, 0.0])  # Agent location
        self.fovea = np.array([0.0, 0.0])  # Fovea location
        self.t = 0

        # Params
        self.fovea_max_translate = 30.0
        self.drag_rate = 70  # Pixels / sec while dragging
        self.attend_rate = 220  # Pixels / sec while moving fovea
        self.failed_reach_secs = 1
        self.fov_jitter = 10
        self.verbose = verbose

        # History
        self.last_failed_reach = None  # tuple (from_id, to_id)
        self.path = [self.loc]
        self.path_ts = [self.t]
        self.path_ids = [env.origin_node_id]
        self.attention = []  # List of dicts{x, y, reachable}

    def init_fovea(self):
        self.move_fovea(np.array([0.0, 0.0]))

    def last_move_direction(self, since_t=0):
        if len(self.path) >= 2:
            last_move_t = self.path_ts[-1]
            if not since_t or last_move_t >= self.t + since_t:
                return self.direction_to(self.path[-2], self.path[-1])

    def current_node(self):
        return self.path_ids[-1]

    def direction_to(self, loc1, loc2):
        return np.arctan2(loc2[1] - loc1[1], loc2[0] - loc1[0])

    def distance_to(self, loc1, loc2=None):
        if loc2 is None:
            loc2 = self.loc
        return np.sqrt((loc1[0] - loc2[0]) ** 2 + (loc1[1] - loc2[1]) ** 2)

    def dist_to_nearest_goal(self):
        g = self.env.nearest_goal(self.loc, r=MAP_W)
        if g:
            return self.distance_to([g.get('x'), g.get('y')], self.loc)

    def reachable_nodes(self):
        points, ids = self.env.nearest_points(self.loc, r=REACH_ZONE_R)
        return ids

    def move_fovea(self, loc, dragging=False):
        """
        If move distance > fovea_max_translate, move in steps of size
        fovea_max_translate.
        """
        from_loc = self.fovea.copy()
        arrived = False
        # Add jitter
        target_loc = loc + np.random.rand(2) * self.fov_jitter - self.fov_jitter/2.
        curr_loc = from_loc if from_loc is not None else target_loc
        full_dist = self.distance_to(from_loc, target_loc)
        while not arrived:
            dist_to_target = self.distance_to(curr_loc, target_loc)
            if dist_to_target <= self.fovea_max_translate:
                curr_loc = target_loc
                arrived = True
            else:
                delta = target_loc - curr_loc
                delta /= np.linalg.norm(delta)  # Normalize
                curr_loc += self.fovea_max_translate * delta
            obs = self.env.observation(curr_loc)
            self.fovea = curr_loc
            self.fovea_moved(curr_loc, obs)
            if not dragging:
                reachable = float(self.distance_to(curr_loc, self.loc)) <= REACH_ZONE_R * REACHABLE_BUFFER
                self.attention.append({'x': curr_loc[0], 'y': curr_loc[1], 'reachable': reachable})
            if self.runner:
                self.runner.maybe_capture_anim_frame()

        self.fovea = curr_loc
        time_elapsed = full_dist / self.drag_rate if dragging else full_dist / self.attend_rate
        return time_elapsed

    def recent_failed_reach_from(self, id):
        if self.last_failed_reach is not None and self.last_failed_reach[0] == id:
            return self.last_failed_reach[1]

    def reach_for_node(self, id):
        goal_id = None
        n = self.env.node(id)
        x, y = n.get('x'), n.get('y')
        n_loc = np.array([x, y]).astype(np.float)
        goal = n.get('type') == 'goal'
        # Move fovea (cursor) to node we're reaching for
        time_for_attempt = self.move_fovea(n_loc)
        dist = self.distance_to(n_loc)
        reachable = dist <= REACH_ZONE_R
        if reachable:
            # Account for drag time to reach target, then drag to center
            time_for_attempt += self.move_fovea(n_loc, dragging=False)
            self.loc = (x, y)
            node_id = n.get('id')
            self.path.append((x, y))
            self.path_ts.append(self.t)
            self.path_ids.append(n.get('id'))
            self.moved_to_node(node_id, (x, y))
            if self.verbose:
                print("(t=%d) Moving to node %s (%.2f, %.2f), distance: %.2f" % (self.t, id, x, y, dist))
        else:
            time_for_attempt += self.failed_reach_secs
            self.last_failed_reach = (self.current_node(), id)
            if self.verbose:
                print("Failed to reach %s from %s, distance=%.2f" % (id, self.current_node(), dist))
        done = goal and reachable
        if done:
            goal_id = n.get('id')
        return done, goal_id, time_for_attempt

    def step(self, t, render=False):
        """
        On each step, agent observes present fovea location, and takes two actions:
        Visual update: move fovea
        Reach: attempt to move agent to node (successful if reachable)
        """
        self.t = t
        node_id, fovea_loc, time_elapsed = self.process_and_act(render=render)
        if fovea_loc is not None:
            time_elapsed += self.move_fovea(fovea_loc)
        done = False
        goal_id = None
        if node_id is not None:
            done, goal_id, reach_time = self.reach_for_node(node_id)
            time_elapsed += reach_time
        if self.runner:
            self.runner.maybe_capture_anim_frame()
        return (done, goal_id, time_elapsed)

    def process_and_act(self, render=False):
        """
        Return tuple:
          node_id: id of node to attempt to move to
          fovea_loc: location (tuple x, y) to move fovea
          time_expired:
        """
        raise NotImplementedError

    def moved_to_node(self, node_id, loc):
        raise NotImplementedError

