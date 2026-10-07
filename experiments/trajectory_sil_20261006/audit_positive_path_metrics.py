"""Reconstruct development SPL from CPU navmesh distances and raw paths.

No renderer, model inference, training, or reserved-screen outputs are used.
This audit is independent of the pair analyzer and compact exporter.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import habitat_sim
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    manifest_path = root / 'prepared_data/development256.json'
    assert sha(manifest_path) == '8e4d2e319b8d88840775bdd7c8173eb8229615222ccf74b10eac65ae56ed52c3'
    manifest = json.loads(manifest_path.read_text())
    ids = [str(x) for x in manifest['episode_ids']]
    assert len(ids) == len(set(ids)) == 256
    dataset_path = root / 'data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz'
    assert sha(dataset_path) == manifest['source_sha256']
    with gzip.open(dataset_path, 'rt') as handle:
        episodes = {str(x['episode_id']): x for x in json.load(handle)['episodes']}
    pathfinders, distances, navmesh_hashes = {}, {}, {}
    info_deltas = []
    for episode_id in ids:
        episode = episodes[episode_id]
        scene = episode['scene_id'].replace('data/scene_datasets/', '', 1)
        if scene not in pathfinders:
            mesh = (root / 'data/scene_datasets' / scene).with_suffix('.navmesh')
            finder = habitat_sim.PathFinder()
            assert finder.load_nav_mesh(str(mesh)), str(mesh)
            pathfinders[scene] = finder
            navmesh_hashes[scene] = sha(mesh)
        goal_distances = []
        for goal in episode['goals']:
            path = habitat_sim.ShortestPath()
            path.requested_start = np.asarray(episode['start_position'], dtype=np.float32)
            path.requested_end = np.asarray(goal['position'], dtype=np.float32)
            assert pathfinders[scene].find_path(path), episode_id
            goal_distances.append(float(path.geodesic_distance))
        distance = min(goal_distances)
        assert math.isfinite(distance) and distance > 0, episode_id
        distances[episode_id] = distance
        info_deltas.append(abs(distance - float(episode['info']['geodesic_distance'])))
    assert len(pathfinders) == 8
    report = {
        'schema': 'positive_pilot_cpu_navmesh_spl_audit_v1',
        'model_calls': 0, 'navigation_actions': 0, 'renderers_created': 0,
        'episodes': 256, 'scenes': 8, 'manifest_sha256': sha(manifest_path),
        'dataset_sha256': sha(dataset_path), 'navmesh_sha256': navmesh_hashes,
        'max_dataset_info_vs_cpu_initial_distance_error_m': max(info_deltas),
        'spl_formula': 'success * initial_geodesic / max(initial_geodesic, logged_path_length)',
        'arms': {},
    }
    for arm in ('control', 'candidate'):
        folder = root / 'runlogs/positive_development256' / f'positive_trajectory_{arm}_64step_seed11'
        rows = [json.loads(p.read_text()) for p in folder.glob('**/stats_*_0.json')]
        assert len(rows) == 256 and {str(x['id']) for x in rows} == set(ids)
        assert all(x.get('early_stop_reason') != 'inference_error' for x in rows)
        errors, spl_values, success_ratios = [], [], []
        for row in rows:
            success, length = float(row['success']), float(row['path_length'])
            distance = distances[str(row['id'])]
            assert success in (0.0, 1.0) and math.isfinite(length) and length >= 0
            if success:
                assert float(row['distance_to_goal']) < 3.000001
                success_ratios.append(length / distance)
            value = success * distance / max(distance, length)
            spl_values.append(value)
            errors.append(abs(value - float(row['spl'])))
        assert max(errors) <= 1e-6, (arm, max(errors))
        report['arms'][arm] = {
            'successes': int(sum(float(x['success']) for x in rows)),
            'successes_with_zero_path_length': sum(bool(x['success']) and float(x['path_length']) == 0 for x in rows),
            'max_success_path_to_initial_distance_ratio': max(success_ratios),
            'mean_raw_spl': sum(float(x['spl']) for x in rows) / len(rows),
            'mean_reconstructed_spl': sum(spl_values) / len(rows),
            'max_raw_vs_reconstructed_spl_error': max(errors),
        }
    report['paired_reconstructed_spl_points'] = 100 * (
        report['arms']['candidate']['mean_reconstructed_spl'] -
        report['arms']['control']['mean_reconstructed_spl'])
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
