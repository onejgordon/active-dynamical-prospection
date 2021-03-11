import json
import pandas as pd
import numpy as np

from agent_adp import ActiveDynamicalProspectionAgent
from environment import Env
from collections import defaultdict
from datetime import datetime
from time import time
import matplotlib.animation as manimation
import matplotlib.pyplot as plt

from constants import MAP_W, MAP_H


class TaskRunner():

    def __init__(self, map_id=1,
                 task_secs_allowed=60.0,
                 agent_kwargs={}, T=0, verbose=False):

        # For anim
        self.writer = None
        self.anim_ax = None

        self.env = Env()
        self.env.load_map(map_id)
        self.agent = ActiveDynamicalProspectionAgent(runner=self, env=self.env, verbose=verbose, **agent_kwargs)
        self.agent.init_fovea()

        self.task_secs_allowed = task_secs_allowed
        self.secs_remaining = self.task_secs_allowed

        self.t = 0  # Steps, not time
        self.T = T  # Max steps (needed?)

    def trial_expired(self):
        return self.secs_remaining <= 0 or self.T and self.t >= self.T

    def elapse_time(self, secs=0.0):
        self.secs_remaining -= secs
        # print("%.2f seconds elapsed, remaining time: %.2f" % (secs, self.secs_remaining))

    def make_anim_fig(self):
        fig, ax = plt.subplots(dpi=200, figsize=(MAP_W/60, MAP_H/60))
        plt.margins(0, 0)
        ax.margins(0, 0)
        ax.set_axis_off()
        ax.axis('off')
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        return fig, ax

    def maybe_capture_anim_frame(self):
        if self.writer:
            self.anim_ax.clear()
            self.agent.render(ax=self.anim_ax)
            self.writer.grab_frame()

    def run(self, render_each=False, render_at_end=True):
        goal_id = None
        while not self.trial_expired():
            done, goal_id, time_elapsed = self.agent.step(self.t, render=render_each)
            self.elapse_time(time_elapsed)
            if done:
                break
            self.t += 1
        result_text = "SUCCESS" if goal_id else "Failed"
        msg = "%s. Steps: %d. " % (result_text, self.t)
        if goal_id:
            msg += "Goal reached: %s" % (goal_id)
        print(msg)
        if render_at_end:
            self.render()
        return goal_id, self.t, self.agent.attention, self.agent.path_ids

    def save_animation(self):
        title = "map_%d_agent_sim" % (self.env.map_id)
        metadata = dict(title=title, artist='JG')
        self.writer = manimation.FFMpegFileWriter(fps=30, metadata=metadata)
        fig, self.anim_ax = self.make_anim_fig()
        fn = "./out/%s.mp4" % (title)
        with self.writer.saving(fig, fn, 144):
            self.run(render_each=False, render_at_end=False)

    def render(self):
        self.agent.render()


class MapBatchRunner():
    def __init__(self, map_id=1, n=20, agent_type='amc', T=0, batch_suffix=None,
                 save_json=True, task_secs_allowed=60.0, agent_kwargs={}):
        self.map_id = map_id
        self.n = n
        self.agent_type = agent_type
        self.task_secs_allowed = task_secs_allowed
        self.T = T
        self.save_json = save_json
        self.agent_kwargs = agent_kwargs
        self.batch_suffix = batch_suffix if batch_suffix is not None else \
            datetime.strftime(datetime.now(), "%Y%m%d%H%M")

    def run(self):
        n_successful = 0
        goal_counts = defaultdict(int)  # Goal_id -> # of reaches
        all_attention = []
        all_paths = []
        all_durations = []
        all_final_dists_to_goal = []
        secs_start = int(time())
        for i in range(self.n):
            print("Running agent simulation %d (map %d)\n=================\n" % (i, self.map_id))
            taskrunner = TaskRunner(map_id=self.map_id, agent_type=self.agent_type,
                                    agent_kwargs=self.agent_kwargs,
                                    task_secs_allowed=self.task_secs_allowed, T=self.T)
            goal_id, t, attention, path = taskrunner.run(render_at_end=False)
            all_attention.extend(attention)
            all_paths.append(path)
            successful = goal_id is not None
            if successful:
                n_successful += 1
                goal_counts[goal_id] += 1
                all_durations.append(taskrunner.task_secs_allowed - taskrunner.secs_remaining)
                all_final_dists_to_goal.append(0)
            else:
                dist = taskrunner.agent.dist_to_nearest_goal()
                if dist is not None:
                    all_final_dists_to_goal.append(dist)

        secs_end = int(time())
        res = {
            'map_id': self.map_id,
            'n': self.n,
            'run_seconds': secs_end - secs_start,
            'agent_kwargs': self.agent_kwargs,
            'T': self.T,
            'agent_type': self.agent_type,
            'pct_successful': n_successful / self.n,
            'goal_counts': goal_counts,
            'all_attention': all_attention,
            'all_paths': all_paths,
            'all_durations': all_durations,
            'mean_final_goal_dist': np.array(all_final_dists_to_goal).mean()
        }
        fn = './out/batch_results_m%d_%s.json' % (self.map_id, self.batch_suffix)
        if self.save_json:
            with open(fn, 'w') as f:
                json.dump(res, f)
            print("Saved %s" % fn)
        return res


class Tuner():

    def __init__(self, n_per_batch=3, base_agent_kwargs={}, task_secs_allowed=60.0, map_id=1, experiments=None):
        """
        experiments (list of tuples): (agent_kwarg_key (str), values (list))
        """
        self.base_agent_kwargs = base_agent_kwargs
        self.map_id = map_id
        self.n_per_batch = n_per_batch
        self.task_secs_allowed = task_secs_allowed
        self.experiments = list(experiments)  # Avoid mutating
        self.results_df = self.build_results_df()

    def build_results_df(self):
        cols = ['map_id', 'n', 'pct_successful', 'mean_task_duration', 'mean_final_goal_dist', 'run_seconds'] \
            + list(self.base_agent_kwargs.keys())
        df = pd.DataFrame(columns=cols)
        return df

    def add_result(self, res):
        row_key = str(len(self.results_df))
        all_durations = res.get('all_durations')
        row = {
            'map_id': res.get('map_id'),
            'n': res.get('n'),
            'pct_successful': res.get('pct_successful'),
            'run_seconds': res.get('run_seconds'),
            'mean_task_duration': np.array(all_durations).mean() if all_durations else -1,
            'mean_final_goal_dist': res.get('mean_final_goal_dist')
        }
        for key, val in res.get('agent_kwargs').items():
            row[key] = val
        self.results_df = self.results_df.append(pd.Series(row, name=row_key))

    def run(self):
        print("Running tuner with base kwargs: %s" % self.base_agent_kwargs)
        self.experiments.append(('base_case', [0]))
        for i, (key, values) in enumerate(self.experiments):
            print("Running tuning experiment %d: %s -> %s" % (i, key, values))
            agent_kwargs = dict(self.base_agent_kwargs)
            for val in values:
                if key != 'base_case':
                    agent_kwargs[key] = val
                print("%s = %s" % (key, val))
                runner = MapBatchRunner(n=self.n_per_batch, map_id=self.map_id,
                                        task_secs_allowed=self.task_secs_allowed,
                                        save_json=False,
                                        agent_kwargs=agent_kwargs)
                res = runner.run()
                self.add_result(res)
        print("Tuning done.")
        # self.results_df.to_pickle("./out/results.pickle")
        return self.results_df
