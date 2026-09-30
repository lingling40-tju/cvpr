"""Pilot turn-level semantic reward wrapper for the existing VLN-CE environment.

This deliberately leaves the actor observation and optimizer unchanged. A turn
may contain up to three actions, so the pilot only assigns turn-level reward.
"""

import math
import json
from functools import lru_cache

import numpy as np
import ray

from vlnce_server.env import VLNCEEnv

from .client import SemanticClient
from .tracker import EventTracker


@lru_cache(maxsize=4)
def load_manifest(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


class SemanticVLNCEEnv(VLNCEEnv):
    def __init__(self, config, simulator, save_video_dir):
        super().__init__(config, simulator, save_video_dir)
        self.semantic_weight = float(config.get("semantic_reward_weight", 0.0))
        self.success_floor = float(config.get("semantic_success_floor", 0.0))
        self.semantic_client = SemanticClient(config.get("semantic_verifier_url", "http://127.0.0.1:5003"))
        self.semantic_parse = {"events": [], "status": "disabled"}
        manifest_path = config.get("semantic_event_manifest", "")
        if self.semantic_weight > 0 and manifest_path:
            record = load_manifest(manifest_path).get(str(config.episode_id))
            if record and record.get("instruction") == self.instruction:
                self.semantic_parse = {"events": record.get("events", []), "status": "curated_manifest"}
            else:
                self.semantic_parse = {"events": [], "status": "manifest_missing_or_mismatch"}
        elif self.semantic_weight > 0:
            try:
                self.semantic_parse = self.semantic_client.parse(self.instruction)
                self.semantic_parse["status"] = "ok"
            except Exception as exc:
                self.semantic_parse = {"events": [], "status": "error", "error": repr(exc)}
        self.tracker = EventTracker(self.semantic_parse.get("events", []), self.semantic_weight)
        self._semantic_eligible = not self.config.history_actions
        self._last_image = None
        self._last_position = None

    def reset(self, seed=None):
        observations, info = super().reset(seed)
        self.tracker = EventTracker(self.semantic_parse.get("events", []), self.semantic_weight)
        self._last_image = observations[-1]["multi_modal_data"]["<image>"][0]
        self._last_position = self._position()
        info.update({
            "semantic_parse": self.semantic_parse,
            "semantic_eligible": self._semantic_eligible,
        })
        return observations, info

    def _position(self):
        locations = ray.get(self.sim.get_locations.remote())["locations"]
        return np.asarray(locations[-1], dtype=float) if len(locations) else np.zeros(3)

    def step(self, response):
        before_image = self._last_image
        before_position = self._last_position
        prior_step = self._current_local_step
        observation, reward, done, info = super().step(response)
        after_image = observation["multi_modal_data"]["<image>"][0]
        after_position = self._position()
        actual_count = self._current_local_step - prior_step
        actions = info.get("actions", [])[:actual_count]
        delta = after_position - before_position
        displacement = float(np.linalg.norm(delta))
        event_before = self.tracker.pending
        verdict = {"status": "skipped", "evidence": "no pending event"}
        semantic_reward = 0.0

        if self._semantic_eligible and event_before and actions:
            try:
                verdict = self.semantic_client.verify(
                    event_before, before_image, after_image, actions,
                    displacement, float(delta[1]),
                )
                semantic_reward = self.tracker.update(verdict.get("status"))
            except Exception as exc:
                verdict = {"status": "error", "evidence": repr(exc)}
        elif not self._semantic_eligible:
            verdict = {"status": "skipped", "evidence": "expert prefix; pilot only scores from scratch"}

        floor_reward = self.success_floor if info.get("task_success") else 0.0
        added_reward = semantic_reward + floor_reward
        reward += added_reward
        if done:
            # The base environment appends its terminal reward inside set_ending_signals.
            self._reward_list[-1] += added_reward
        else:
            self._reward_list.append(reward)

        info["total_reward"] = sum(self._reward_list)
        info["semantic_events"] = self.tracker.events
        info["semantic_event_before"] = event_before
        info["semantic_verdict"] = verdict
        info["semantic_progress"] = self.tracker.progress
        info["semantic_eligible"] = self._semantic_eligible
        info["executed_actions"] = actions
        info["semantic_displacement_m"] = displacement
        info["semantic_vertical_delta_m"] = float(delta[1])
        info["reward_components"]["semantic_reward"] = semantic_reward
        info["reward_components"]["success_floor"] = floor_reward
        info["gen_traj"][-1]["reward"] = reward
        info["gen_traj"][-1]["executed_actions"] = actions
        info["gen_traj"][-1]["semantic_verdict"] = verdict
        info["gen_traj"][-1]["semantic_progress"] = self.tracker.progress
        self._last_image = after_image
        self._last_position = after_position
        assert math.isfinite(reward)
        return observation, reward, done, info
