"""
Dense reward shaping for the Franka-Kitchen tasks.

From-scratch RL (SAC/RLPD via ``run_scratch.py``) only ever sees the *sparse*
kitchen reward: +1 the moment an appliance crosses its completion threshold.
A randomly-initialized policy essentially never reaches that, so the reward is
0.0000 forever, the soft-Q value function has nothing to anchor to, and training
just drifts (ever-growing actor loss, spiking critic loss) without solving
anything.

This wrapper adds a *dense* term that pulls the robot toward the goal
configuration of each selected appliance, using the exact same per-element
distance d4rl uses to decide completion::

    distance(element) = || obs[OBS_ELEMENT_INDICES[element]]
                          - OBS_ELEMENT_GOALS[element] ||

The raw kitchen observation is ``[qp(9), obj_qp(21), goal(30)]`` (60-dim), so the
current joint/object positions live in ``obs[0:30]`` and are indexed directly by
``OBS_ELEMENT_INDICES``; the goals are constants.

Two shaping modes:

* **plain** (default) -- ``shaped = sparse - scale * sum_e distance(e, s')``.
  A strong, simple gradient toward every selected goal. Most reliable way to get
  a usable signal from scratch. Its optimum (all distances -> 0) coincides with
  completing every selected task, so the bias is benign here.
* **potential_based** -- ``shaped = sparse + scale * (gamma*Phi(s') - Phi(s))``
  with ``Phi(s) = -sum_e distance(e, s)``. Ng et al. (1999) potential-based
  shaping: provably preserves the optimal policy (no bias), but the per-step
  difference is a weaker signal.

IMPORTANT: this must wrap the *raw* d4rl kitchen env, i.e. be the **innermost**
wrapper (listed first in the ``wrappers`` config, before
``mujoco_locomotion_lowdim``), because it reads raw, unnormalized positions.
"""

import os
import json
import logging

import numpy as np
import gym

log = logging.getLogger(__name__)

# Raw obs layout: [qp(9), obj_qp(21), goal(30)]. Positions occupy obs[0:N_POS]
# and are indexed directly by OBS_ELEMENT_INDICES (values 0..29).
N_POS = 30

# Maps a kitchen task element to the mujoco *site* marking that appliance, so we
# can shape an end-effector "reach" term toward it. Only the appliances whose
# site is unambiguous are listed; burners (which knob maps to which burner is
# ambiguous) are intentionally omitted -- they simply get no reach term.
ELEMENT_TO_SITE = {
    "light switch": "light_site",
    "slide cabinet": "slide_site",
    "hinge cabinet": "hinge_site1",
    "microwave": "microhandle_site",
    "kettle": "kettle_site",
}
EE_SITE = "end_effector"


class KitchenDenseRewardWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        task_elements=None,
        dense_reward_scale=1.0,
        reach_reward_scale=0.0,
        potential_based=False,
        gamma=0.99,
    ):
        super().__init__(env)
        # d4rl is imported lazily so importing the wrapper registry never forces
        # a d4rl.kitchen import in robomimic/furniture-only setups.
        from d4rl.kitchen.kitchen_envs import (
            OBS_ELEMENT_INDICES,
            OBS_ELEMENT_GOALS,
        )

        self.dense_reward_scale = float(dense_reward_scale)
        self.reach_reward_scale = float(reach_reward_scale)
        self.potential_based = bool(potential_based)
        self.gamma = float(gamma)

        tasks = self._resolve_tasks(task_elements)

        # Precompute (indices, goal) per selected element once.
        self._element_targets = []
        for element in tasks:
            if element not in OBS_ELEMENT_INDICES:
                raise ValueError(
                    f"Unknown kitchen task element '{element}'. "
                    f"Valid: {sorted(OBS_ELEMENT_INDICES)}"
                )
            self._element_targets.append(
                (
                    np.asarray(OBS_ELEMENT_INDICES[element], dtype=int),
                    np.asarray(OBS_ELEMENT_GOALS[element], dtype=np.float64),
                )
            )
        self._tasks = list(tasks)
        self._prev_potential = None

        # Resolve mujoco site ids for the end-effector "reach" term. The object
        # distance term (above) gives no gradient until the arm already touches
        # the appliance, so from-scratch exploration stalls; this -||ee - site||
        # term pulls the arm to the appliance first. Only active when
        # reach_reward_scale > 0.
        self._ee_site_id = None
        self._reach_site_ids = []
        if self.reach_reward_scale:
            sim = getattr(self.env.unwrapped, "sim", None)
            if sim is None:
                log.warning(
                    "reach_reward_scale set but env has no .sim; disabling reach term."
                )
                self.reach_reward_scale = 0.0
            else:
                self._ee_site_id = sim.model.site_name2id(EE_SITE)
                for element in self._tasks:
                    site = ELEMENT_TO_SITE.get(element)
                    if site is None:
                        log.warning(
                            "No reach site mapped for '%s'; skipping its reach term.",
                            element,
                        )
                        continue
                    self._reach_site_ids.append(sim.model.site_name2id(site))

        log.info(
            "KitchenDenseRewardWrapper: shaping %d task(s) %s "
            "(scale=%.3g, reach_scale=%.3g, reach_sites=%d, potential_based=%s)",
            len(self._tasks),
            self._tasks,
            self.dense_reward_scale,
            self.reach_reward_scale,
            len(self._reach_site_ids),
            self.potential_based,
        )

    def _resolve_tasks(self, task_elements):
        """Mirror script/run.py: explicit arg > DPPO_KITCHEN_TASKS env var >
        whatever the underlying env is configured to reward."""
        if task_elements:
            return list(task_elements)
        raw = os.environ.get("DPPO_KITCHEN_TASKS")
        if raw:
            return list(json.loads(raw))
        return list(getattr(self.env.unwrapped, "TASK_ELEMENTS", []))

    @staticmethod
    def _as_array(obs):
        """reset/step may return obs or (obs, info) depending on gym version."""
        return obs[0] if isinstance(obs, tuple) else obs

    def _reach_distance(self):
        """sum_e ||ee_xpos - site_e_xpos||  (Cartesian arm-to-appliance distance,
        from the live mujoco sim). 0.0 when the reach term is disabled."""
        if not self._reach_site_ids:
            return 0.0
        data = self.env.unwrapped.sim.data
        ee = data.site_xpos[self._ee_site_id]
        total = 0.0
        for site_id in self._reach_site_ids:
            total += float(np.linalg.norm(ee - data.site_xpos[site_id]))
        return total

    def _potential(self, obs):
        """Phi(s) = -[ dense_reward_scale * sum_e ||pos_e - goal_e||
                       + reach_reward_scale * sum_e ||ee - site_e|| ]
        Scales are folded in so both plain and potential-based modes use it
        directly. Maximal (0) only when every appliance is at its goal (and, if
        the reach term is on, the arm is at the appliance)."""
        obs = np.asarray(obs, dtype=np.float64)
        obj = 0.0
        for idx, goal in self._element_targets:
            obj += float(np.linalg.norm(obs[idx] - goal))
        pot = -self.dense_reward_scale * obj
        if self.reach_reward_scale:
            pot -= self.reach_reward_scale * self._reach_distance()
        return pot

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._prev_potential = self._potential(self._as_array(obs))
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        potential = self._potential(self._as_array(obs))

        if self.potential_based:
            prev = self._prev_potential
            if prev is None:
                prev = potential
            dense = self.gamma * potential - prev
        else:
            dense = potential  # scales already folded into the potential

        self._prev_potential = potential

        info = dict(info) if info is not None else {}
        info["sparse_reward"] = float(reward)
        info["dense_reward"] = float(dense)
        return obs, float(reward) + dense, done, info
