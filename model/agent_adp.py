from agent import Agent
import util

from collections import defaultdict
from recordclass import recordclass
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import cm
from scipy.stats import circstd
from constants import REACH_ZONE_R, MAP_W, MAP_H, NODE_R, GOAL_R, \
                      FOV_R
import numpy as np


class ActiveDynamicalProspectionAgent(Agent):

    def __init__(self, **kwargs):
        super(ActiveDynamicalProspectionAgent, self).__init__(**kwargs)
        self.jitter_radius = 20
        self.decay_rate = 0.98
        self.n_particles_step = kwargs.get('n_particles_step', 20)  # Particles emitted per step
        self.particle_mass = kwargs.get('particle_mass', 2)
        self.particle_discount = kwargs.get('particle_discount', 0.2)
        self.sample_temperature = kwargs.get('sample_temperature', 0.05)
        self.init_goal_energy = kwargs.get('init_goal_energy', 0.2)
        self.learn_alpha = kwargs.get('learn_alpha', 0.1)
        self.hm_origin_energy = kwargs.get('hm_origin_energy', 0.6)  # Prior at origin
        self.init_hm_energy = kwargs.get('init_hm_energy', 0.2)
        self.policy_crit = kwargs.get('policy_crit', 'theta_var')
        self.step_conf_thresh = kwargs.get('step_conf_thresh', 0.7)
        self.node_exit_energy = kwargs.get('node_exit_energy', 'naive')

        # View heatmap (track agent's visual search at particular resolution)
        self.heatmap_coords = None
        self.heatmap = None  # Low to high energy (0 - 1)
        self.heatmap_prior = None
        self.heatmap_ground = None
        HM_W = kwargs.get('hm_w', 100)
        self.heatmap_res = (HM_W, int(MAP_W/MAP_H * HM_W))
        self.init_heatmap()

        self.Particle = recordclass('Particle', 'x y momentum theta')
        self.Trajectory = recordclass('Trajectory', 'xs ys ms es')
        self.step_trajs = []  # Temporary storage of particle trajectories for step (for render & reach failure)

    def init_heatmap(self):
        """
        Init heatmap with gradient/gravity towards goals
        """
        self.heatmap = np.ones(self.heatmap_res) * self.init_hm_energy
        self.heatmap_prior = np.zeros(self.heatmap_res)

        hm_w, hm_h = self.heatmap_res
        self.heatmap_coords = np.meshgrid(
            np.arange(hm_h),
            np.arange(hm_w)
        )
        # Attractors around rewards (with lowest energy filling goal radii)
        dist_layers = []
        for g in self.env.goals():
            x, y = g.get('x'), g.get('y')
            hm_x, hm_y = self.hm_loc(x, y)
            xx, yy = self.heatmap_coords
            dist = np.sqrt((xx - hm_x) ** 2 + (yy - hm_y) ** 2) + 50
            g_mask = self.circle_hm_mask(x, y, r=GOAL_R)
            dist *= 1 - g_mask
            dist_layers.append(dist)
        self.heatmap_prior = np.min(np.stack(dist_layers, axis=2), axis=2)

        # Normalize to hm_origin_energy at start - init_goal_energy at goals
        tgt_range = self.hm_origin_energy - self.init_goal_energy
        hm_x, hm_y = self.hm_loc(self.loc[0], self.loc[1])
        curr_range = self.energy_at(*self.loc, prior_only=True) - self.heatmap_prior.min()
        scale = tgt_range / curr_range
        self.heatmap_prior *= scale
        self.heatmap_prior += self.init_goal_energy - self.heatmap_prior.min()

        self.heatmap = self.naive_heatmap()

    def naive_heatmap(self):
        """
        Heatmap initial conditions, towards which to decay
        """
        return np.clip(self.heatmap_prior + self.init_hm_energy, 0, 1)

    def hm_loc(self, x, y, pixel_correction=0):
        hm_h, hm_w = self.heatmap.shape
        hm_x = int((x + MAP_W/2 - 1)/MAP_W * hm_w) + pixel_correction
        hm_y = int((y + MAP_H/2 - 1)/MAP_H * hm_h) + pixel_correction
        return hm_x, hm_y

    def real_loc(self, hm_x, hm_y):
        hm_h, hm_w = self.heatmap.shape
        return (hm_x + 0.5)/(hm_w) * MAP_W - MAP_W/2, (hm_y + 0.5)/(hm_h) * MAP_H - MAP_H/2

    def heatmap_composite(self):
        """
        Mean of prior and heatmap -- range stil in [0, 1]

        If self.last_failed_reach is from current node, mask heatmap at
        failed reach target as temporary inhibition for rollouts
        """
        failed_target_id = self.recent_failed_reach_from(self.current_node())
        failed_mask = None
        if failed_target_id is not None:
            n = self.env.node(failed_target_id)
            r = GOAL_R if n.get('type') == 'goal' else NODE_R
            failed_mask = self.circle_hm_mask(n.get('x'), n.get('y'), r=1.1 * r)
        hmc = np.maximum(self.heatmap, self.heatmap_prior)
        if failed_mask is not None:
            hmc += failed_mask
        return np.clip(hmc, 0, 1)

    def energy_at(self, x, y, prior_only=False, heatmap_only=False):
        hm_x, hm_y = self.hm_loc(x, y)
        if prior_only:
            hm = self.heatmap_prior
        elif heatmap_only:
            hm = self.heatmap
        else:
            # Composite
            hm = self.heatmap_composite()
        h, w = hm.shape
        if hm_x >= w or hm_x < 0 or hm_y >= h or hm_y < 0:
            # Out of bounds, return max heat (1.0)
            return 1.0
        return hm[hm_y, hm_x]

    def min_energy_loc(self, hm, mask=None, default=None):
        """
        """
        if mask is not None:
            hm = (hm + (1-mask)).clip(0, 1)
        if hm.min() == 1:
            # Flat surface
            loc = default
        else:
            idx = np.argmin(hm)
            xx, yy = self.heatmap_coords
            hm_x, hm_y = xx.flatten()[idx], yy.flatten()[idx]
            loc = self.real_loc(hm_x, hm_y)
        return loc

    def softmax_sample(self, hm, mask=None, temperature=None, debug=False):
        """
        Sample location from heatmap as probability distribution

        temperature: Parameter for softmax
        """
        weights = None
        if mask is not None:
            hm[~mask] = 1  # (hm + 1 - mask).clip(0, 1)

        # Treat temperature as temperature of softmax
        masked_hm = hm[mask].flatten()
        if not len(masked_hm):
            return
        weights = util.softmax(1-masked_hm, temperature=temperature if temperature else 1.0)
        masked_pixel = np.random.choice(np.arange(masked_hm.size), p=weights)
        pixel = np.where(mask.flatten())[0][masked_pixel]

        xx, yy = self.heatmap_coords
        hm_x, hm_y = xx.flatten()[pixel], yy.flatten()[pixel]
        loc = self.real_loc(hm_x, hm_y)
        if debug:
            print("Chose %s (hm: %d, %d)" % (loc, hm_x, hm_y))
            n_charts = 2 if weights is not None else 1
            fig, axs = plt.subplots(1, n_charts, dpi=144)
            fig.axes[0].imshow(hm, origin='lower')
            fig.axes[0].scatter(hm_x, hm_y, color='red', s=4)
            if weights is not None:
                im = fig.axes[1].imshow(weights.reshape(*hm.shape), origin='lower')
                fig.axes[1].scatter(hm_x, hm_y, color='red', s=4)
                fig.colorbar(im)
            plt.show()
        return loc

    def circle_hm_mask(self, x, y, r=NODE_R, direction=None, spread=3*np.pi/2, binary=False):
        hm_h, hm_w = self.heatmap.shape
        hm_x, hm_y = self.hm_loc(x, y, pixel_correction=0.5)
        fov_r_hm = r/MAP_W * hm_w
        yy, xx = np.ogrid[:hm_h, :hm_w]
        dist = ((xx - hm_x)**2 + (yy-hm_y)**2)**0.5
        mask = 1 - (dist - fov_r_hm).clip(0, 1)
        if direction is not None:
            theta_field = np.arctan2(yy - hm_y, xx - hm_x)
            dtheta_field = np.abs(np.arctan2(np.sin(theta_field-direction), np.cos(theta_field-direction)))
            dir_mask = spread/2 < dtheta_field
            mask[dir_mask] = 0
        if binary:
            mask = mask > 0.5
        return mask

    def ring_hm_mask(self, x, y, r=NODE_R, r_inner=0, direction=None, spread=3*np.pi/2,
                     binary=False):
        """
        If direction is supplied, restrict ring to 3pi/2 radians centered on direction.
        """
        mask_outer = self.circle_hm_mask(x, y, r=r, direction=direction, spread=spread)
        if r_inner:
            # Ring mask
            mask_inner = self.circle_hm_mask(x, y, r=r_inner, direction=direction,
                                             spread=spread)
            mask = mask_outer - mask_inner
        else:
            mask = mask_outer
        if binary:
            mask = mask > 0.5
        return mask

    def particle_move(self, p):
        """
        Dynamics to consider:
            - Momentum (via p.theta)
            - Bias to existing gradient (done via sampling?)
        """
        # Make heatmap via masks
        hm = self.heatmap_composite()
        mask = self.ring_hm_mask(p.x, p.y,
                                 r=REACH_ZONE_R,
                                 r_inner=REACH_ZONE_R/4,
                                 direction=p.theta,
                                 spread=np.pi,
                                 binary=True)
        origin_e = self.energy_at(p.x, p.y)
        dest = self.softmax_sample(hm, mask=mask, temperature=self.sample_temperature, debug=False)
        if dest is None:
            return p, True
        dest_e = self.energy_at(dest[0], dest[1])
        same_loc = p.x == dest[0] and p.y == dest[1]
        very_near_goal = self.env.is_near_goal(dest)
        terminate = same_loc or very_near_goal
        new_momentum = p.momentum - dest_e/self.particle_mass
        new_momentum += - 2 * (dest_e - origin_e) / self.particle_mass
        p.momentum = max([0, new_momentum])
        p.theta = self.direction_to([p.x, p.y], dest)
        p.x, p.y = dest
        return p, terminate

    def rollout_particle(self, p, max_steps=10, min_energy=0.1):
        """
        MC roll-out of a trajectory, and at each landing,
        reduce heatmap energy (weight by energy?)
        """
        i = 0
        done = False
        first_step = None
        traj = self.Trajectory([p.x], [p.y], [p.momentum], [self.energy_at(p.x, p.y)])
        from_agent = self.loc[0] == p.x and self.loc[1] == p.y
        while not done:
            p, terminate = self.particle_move(p)
            is_first_step = from_agent and i == 0
            if is_first_step:
                # Determine if first step of rollout falls on a node
                n = self.env.is_within_node(np.array([p.x, p.y]), lenience=2.0)
                first_step = ([p.x, p.y], n.get('id') if n else None)
            i += 1
            done = terminate or p.momentum <= min_energy or i > max_steps
            traj.xs.append(p.x)
            traj.ys.append(p.y)
            traj.ms.append(p.momentum)
            traj.es.append(self.energy_at(p.x, p.y))
        final_energy, _ = traj.es[-1], traj.ms[-1]
        hm = self.heatmap_composite()
        for i, (x, y, momentum, energy) in enumerate(zip(traj.xs, traj.ys, traj.ms, traj.es)):
            if i == 0:
                continue  # Don't update heat at origin of rollout
            # Reduce energy under particle at each step, discounted by momentum
            mask = self.circle_hm_mask(x, y, r=NODE_R)
            dheat = final_energy - hm
            discount = momentum * self.particle_discount  # pd -> 0
            self.heatmap += mask * (1 - discount) * self.learn_alpha * dheat
        self.heatmap = np.clip(self.heatmap, 0, 1)
        return traj, first_step

    def emit_particles(self, force_origin=None):
        """
        Emit particle simulations given present fovea observations
        Obs are a transient overlay on present heatmap used for simulations
        such that not all observed holds become stored in the model, only
        those used by particles.
        """
        all_first_steps = []  # Store (loc, node ID) of first step (for rollouts from agent loc)
        self.step_trajs = []
        for i in range(self.n_particles_step):
            ptheta = None
            if force_origin is not None:
                ox, oy = force_origin
            else:
                ox, oy = self.loc
                ptheta = self.last_move_direction(since_t=-1)
            if self.env.is_near_goal(np.array([ox, oy])):
                # Don't emit from goal
                continue
            # Particle direction starts with agent prior move direction (if originating from agent)
            p = self.Particle(ox, oy, 1, ptheta)
            traj, first_step = self.rollout_particle(p)
            if self.runner:
                self.runner.maybe_capture_anim_frame()
            all_first_steps.append(first_step)
            self.step_trajs.append(traj)
        return all_first_steps

    def decide_to_navigate(self, all_first_steps):
        """
        Using either node pluraility of first steps, or theta standard deviation,
        decide whether to move to a nearby node
        """
        go_to = plur_node = None
        n = len(all_first_steps)
        if not n:
            return
        counts = defaultdict(int)
        for (loc, node_id) in all_first_steps:
            if node_id is not None:
                counts[node_id] += 1
        if len(counts):
            plur_node = max(counts, key=counts.get)
            pct = counts[plur_node]/n
            if self.policy_crit == "node_plurality":
                # step_conf_thresh is plurality percentage threshold
                if plur_node is not None and pct >= self.step_conf_thresh:
                    go_to = plur_node
            elif self.policy_crit == "theta_var":
                # step_conf_thresh is std(theta) threshold (rad)
                thetas = [self.direction_to(step[0], self.loc) for step in all_first_steps]
                std = circstd(thetas)
                if std < self.step_conf_thresh:  # Interpreted as max std
                    go_to = plur_node
        return go_to

    def moved_to_node(self, node_id, loc):
        # Increase energy at prior node
        if len(self.path) >= 2:
            prev_loc = self.path[-2]
            prev_mask = self.circle_hm_mask(prev_loc[0], prev_loc[1], r=NODE_R * 1.1, binary=True)
            if self.node_exit_energy == "naive":
                self.heatmap[prev_mask] = self.naive_heatmap()[prev_mask]
            elif self.node_exit_energy == "max":
                self.heatmap[prev_mask] = 1.0

    def error_map(self):
        """
        Generate error map based on heat levels along recent trajectories
        """
        emap = np.zeros(self.heatmap_res)
        for traj in self.step_trajs:
            for x, y, h in zip(traj.xs, traj.ys, traj.es):
                emap[self.circle_hm_mask(x, y, r=NODE_R, binary=True)] = h
        return emap

    def choose_fovea_loc(self, render=False):
        return self.maximum_heat(render=render)

    def maximum_heat(self, step_size=10.0, render=False):
        """
        Return foveal location of maximum (mean) heat of the error map
        """
        emap = self.error_map()
        hm_w, hm_h = self.heatmap.shape
        max_heat, max_heat_loc = None, (None, None)
        for x in np.arange(-MAP_W/2, MAP_W/2, step_size):
            for y in np.arange(-MAP_H/2, MAP_H/2, step_size):
                heat = emap[self.circle_hm_mask(x, y, r=FOV_R, binary=True)].sum()
                if max_heat is None or heat > max_heat:
                    max_heat = heat
                    max_heat_loc = np.array([x, y])
        if render:
            r = FOV_R / MAP_W * hm_w
            fig, ax = plt.subplots(dpi=144)
            ax.imshow(emap, origin='lower', vmin=0)
            ax.add_patch(Circle(self.hm_loc(max_heat_loc[0], max_heat_loc[1]), r, fill=False, color='red'))
            plt.show()
        return max_heat_loc

    def fovea_moved(self, loc, obs=None):
        """
        Update heatmap during (or after) saccade

        obs: List(node)
        """
        fov_mask = self.circle_hm_mask(loc[0], loc[1], r=FOV_R)
        obs_mask = np.zeros(self.heatmap.shape, dtype=bool)
        if obs is not None:
            for n in obs:
                x, y = n.get('x'), n.get('y')
                r = GOAL_R if n.get('type') == 'goal' else NODE_R
                node_mask = self.circle_hm_mask(x, y, r=r, binary=True)  # r*1.3
                # fov_mask -= node_mask
                obs_mask += node_mask
        # Increase energy to max for all in foveal area
        # Decrease energy to min(heatmap, (naive + prior)/2) under observed nodes
        self.heatmap += fov_mask
        midway = (self.naive_heatmap() + self.heatmap_prior) / 2.
        self.heatmap[obs_mask] = np.minimum(self.heatmap[obs_mask], midway[obs_mask])
        self.heatmap = np.clip(self.heatmap, 0, 1)

    def decay_heatmap(self):
        """
        Decay heatmap towards naive (initial/naive)
        """
        naive = self.naive_heatmap()
        self.heatmap += (naive - self.heatmap) * (1-self.decay_rate)

    def process_and_act(self, render=False):
        time_processing = 0.2
        all_first_steps = self.emit_particles()
        fovea_loc = self.choose_fovea_loc(render=False)
        node_id = self.decide_to_navigate(all_first_steps)
        if render:
            self.render()
        self.decay_heatmap()
        return (node_id, fovea_loc, time_processing)

    def render(self, render_nodes=True, with_rollouts=True, ax=None):
        TRAJ_CMAP = cm.get_cmap("hot")
        ax = self.env.render_map(agent_loc=self.loc,
                                 fovea_loc=self.fovea,
                                 render_nodes=render_nodes,
                                 path=self.path, ax=ax)
        im = ax.imshow(self.heatmap_composite(),
                       extent=(-MAP_W/2, MAP_W/2, -MAP_H/2, MAP_H/2), alpha=0.3,
                       vmin=0, vmax=1,
                       cmap='plasma', origin='lower')
        if with_rollouts and self.step_trajs:
            for traj in self.step_trajs:
                cmap = matplotlib.cm.get_cmap(TRAJ_CMAP)
                ax.plot(traj.xs, traj.ys, linewidth=0.8, c=cmap(0.5), alpha=0.3)
                ax.scatter(traj.xs, traj.ys, s=3, c=TRAJ_CMAP(traj.ms))
        ax.text(-MAP_W/2 + 20, MAP_H/2 - 20, "M%d, t: %d" % (self.env.map_id, self.t), c='black')
        return ax
