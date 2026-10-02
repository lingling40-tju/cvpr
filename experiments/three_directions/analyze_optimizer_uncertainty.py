"""Exploratory scene and seed uncertainty from a verified optimizer package.

The 1,839-episode comparison is primary. The 256-episode development screen
and its 1,583-episode complement are reported separately to expose selection.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from verify_optimizer_scaled_package import check_result, difference


SEEDS = (11, 22, 33)


def interval(draws: list[float]) -> list[float]:
    ordered = sorted(draws)
    return [ordered[int(len(ordered) * 0.025)], ordered[int(len(ordered) * 0.975)]]


def analyze_subset(rows: dict, ids: list[str], scenes: list[str],
                   draws: int, random_seed: int) -> dict:
    assert len(ids) == len(scenes) and len(ids) == len(set(ids)) and draws > 0
    scene_ids = defaultdict(list)
    for episode, scene in zip(ids, scenes):
        scene_ids[scene].append(episode)
    scene_names = sorted(scene_ids)
    assert len(scene_names) > 1 and set(rows) == set(SEEDS)
    counts = {scene: len(scene_ids[scene]) for scene in scene_names}
    sums = {}
    for seed in SEEDS:
        sums[seed] = {}
        for scene in scene_names:
            selected = [rows[seed][episode] for episode in scene_ids[scene]]
            sums[seed][scene] = (
                sum(item['candidate']['success'] - item['control']['success']
                    for item in selected),
                sum(float(item['candidate']['spl']) - float(item['control']['spl'])
                    for item in selected),
            )
    paired = {seed: difference(rows[seed], ids) for seed in SEEDS}
    rng = random.Random(random_seed)
    seed_draws = {seed: {'sr_pp': [], 'spl_pp': []} for seed in SEEDS}
    combined = {'sr_pp': [], 'spl_pp': []}
    for _ in range(draws):
        sampled_scenes = [rng.choice(scene_names) for _ in scene_names]
        sampled_seeds = [rng.choice(SEEDS) for _ in SEEDS]
        denominator = sum(counts[scene] for scene in sampled_scenes)
        for seed in SEEDS:
            sr = sum(sums[seed][scene][0] for scene in sampled_scenes)
            spl = sum(sums[seed][scene][1] for scene in sampled_scenes)
            seed_draws[seed]['sr_pp'].append(100 * sr / denominator)
            seed_draws[seed]['spl_pp'].append(100 * spl / denominator)
        combined_denominator = denominator * len(sampled_seeds)
        combined['sr_pp'].append(100 * sum(sums[seed][scene][0]
                                             for seed in sampled_seeds
                                             for scene in sampled_scenes) /
                                  combined_denominator)
        combined['spl_pp'].append(100 * sum(sums[seed][scene][1]
                                              for seed in sampled_seeds
                                              for scene in sampled_scenes) /
                                   combined_denominator)
    return {
        'episodes': len(ids), 'scenes': len(scene_names),
        'paired_seed_differences_pp': {str(seed): {
            metric: paired[seed][metric] for metric in ('sr_pp', 'spl_pp')
        } for seed in SEEDS},
        'mean_paired_difference_pp': {
            metric: statistics.mean(paired[seed][metric] for seed in SEEDS)
            for metric in ('sr_pp', 'spl_pp')
        },
        'per_seed_scene_cluster_95_pp': {str(seed): {
            metric: interval(seed_draws[seed][metric])
            for metric in ('sr_pp', 'spl_pp')
        } for seed in SEEDS},
        'scene_and_seed_cluster_95_pp': {
            metric: interval(combined[metric]) for metric in ('sr_pp', 'spl_pp')
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('package_dir', type=Path)
    parser.add_argument('--mode', choices=('group4', 'kl_anchor', 'dynamic'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--draws', type=int, default=10000)
    args = parser.parse_args()
    assert args.draws >= 1000
    root = args.package_dir
    screen, screen_ids, screen_scenes = check_result(root, args.mode, 256)
    full, full_ids, full_scenes = check_result(root, args.mode, 1839)
    screen_set = set(screen_ids)
    assert screen_set < set(full_ids)
    outside = [(episode, scene) for episode, scene in zip(full_ids, full_scenes)
               if episode not in screen_set]
    assert len(outside) == 1583
    result = {
        'mode': args.mode, 'split': 'val_unseen',
        'draws': args.draws, 'random_seed': 20261003,
        'resampling': 'paired scene clusters and three training seeds, each with replacement',
        'interpretation': 'Exploratory intervals over 11 held-out scenes and only three '
                          'training seeds; the full 1,839 episodes are the primary comparison.',
        'subsets': {
            'development_screen_256': analyze_subset(
                screen, screen_ids, screen_scenes, args.draws, 20261003),
            'full_1839': analyze_subset(
                full, full_ids, full_scenes, args.draws, 20261004),
            'outside_screen_1583': analyze_subset(
                full, [episode for episode, _ in outside],
                [scene for _, scene in outside], args.draws, 20261005),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({name: subset['mean_paired_difference_pp']
                      for name, subset in result['subsets'].items()}, indent=2))


if __name__ == '__main__':
    main()
