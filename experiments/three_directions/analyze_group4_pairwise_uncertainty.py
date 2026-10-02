"""Exploratory scene/seed intervals for both 2+2 GRPO comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_optimizer_uncertainty import analyze_subset
from verify_group4_pairwise_scaled_package import verify_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('package_dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--draws', type=int, default=10000)
    args = parser.parse_args()
    assert args.draws >= 1000
    screen, screen_ids, screen_scenes = verify_count(args.package_dir, 256)
    full, full_ids, full_scenes = verify_count(args.package_dir, 1839)
    screen_set = set(screen_ids)
    assert screen_set < set(full_ids)
    outside = [(episode, scene) for episode, scene in zip(full_ids, full_scenes)
               if episode not in screen_set]
    assert len(outside) == 1583
    comparisons = {}
    for comparison in ('branch_control', 'group4'):
        comparisons[comparison] = {
            'development_screen_256': analyze_subset(
                screen[comparison], screen_ids, screen_scenes, args.draws, 20261003),
            'full_1839': analyze_subset(
                full[comparison], full_ids, full_scenes, args.draws, 20261004),
            'outside_screen_1583': analyze_subset(
                full[comparison], [episode for episode, _ in outside],
                [scene for _, scene in outside], args.draws, 20261005),
        }
    output = {
        'mode': 'group4_pairwise', 'split': 'val_unseen',
        'draws': args.draws,
        'resampling': 'paired scene clusters and three training seeds, each with replacement',
        'interpretation': 'Exploratory intervals over 11 held-out scenes and only three '
                          'training seeds; the full 1,839 episodes are primary.',
        'comparisons': comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({comparison: sections['full_1839']['mean_paired_difference_pp']
                      for comparison, sections in comparisons.items()}, indent=2))


if __name__ == '__main__':
    main()
