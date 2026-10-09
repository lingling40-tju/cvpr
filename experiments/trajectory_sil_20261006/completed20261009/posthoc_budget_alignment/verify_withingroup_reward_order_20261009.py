"""Independent pair/set arithmetic against the original compact training records."""
import argparse
from collections import Counter,defaultdict
import hashlib,json,itertools
from pathlib import Path


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    report=json.loads(a.report.read_text())
    assert sha(a.source)==report['source_compact_sha256']=='62d6576418ab324fac86cd8127ebf2b1cbd155340fd8b4b94497190fd460c790'
    groups=defaultdict(dict)
    for line in a.source.open():
        x=json.loads(line);key=(x['run'],x['step'],x['episode_id'])
        assert x['trajectory_index'] not in groups[key]
        groups[key][x['trajectory_index']]=x
    counts=defaultdict(Counter);all_pairs=set()
    for (run,step,eid),rows in groups.items():
        assert len(rows)==4
        c=counts[run];c['groups']+=1;c['trajectories']+=len(rows)
        near_ids={i for i,x in rows.items() if x['distance_to_goal'] is not None and x['distance_to_goal']<3 and x['end_reason'].startswith('number of ') and x['end_reason'].endswith('exceeded.')}
        far_ids={i for i,x in rows.items() if x['distance_to_goal'] is not None and x['distance_to_goal']>3 and x['end_reason']=='stopped but goal not reached.' and x['ndtw_reward']>0}
        c['near_timeout_trajectories']+=len(near_ids)
        for i in near_ids:
            other_sum=sum(x['total_reward'] for j,x in rows.items() if j!=i)
            c['near_timeouts_below_group_mean']+=3*rows[i]['total_reward']<other_sum
            c['near_timeouts_equal_group_mean']+=3*rows[i]['total_reward']==other_sum
        c['groups_with_near_timeout']+=len(near_ids)>0
        c['groups_with_near_timeout_and_far_failed_stop']+=len(near_ids)>0 and len(far_ids)>0
        c['groups_with_no_logged_success']+=all(x['task_success'] is False for x in rows.values())
        c['no_success_groups_with_positive_failed_stop']+=all(x['task_success'] is False for x in rows.values()) and len(far_ids)>0
        c['nonfinite_distance_records']+=len([x for x in rows.values() if x['distance_to_goal'] is None])
        for left,right in itertools.combinations(rows,2):
            for ni,fi in [(left,right),(right,left)]:
                if ni in near_ids and fi in far_ids and rows[ni]['total_reward']<rows[fi]['total_reward']:
                    assert rows[ni]['distance_to_goal']<rows[fi]['distance_to_goal']
                    all_pairs.add((run,step,eid,ni,fi))
                    c['far_failed_stop_over_near_timeout_pairs']+=1
    recorded=set();case_keys=set()
    for case in report['rank_conflict_cases']:
        key=(case['run'],case['step'],case['episode_id'])
        assert key not in case_keys;case_keys.add(key)
        for pair in case['pairs']:
            ni,fi=pair['near_trajectory_index'],pair['far_trajectory_index']
            item=key+(ni,fi);assert item not in recorded;recorded.add(item)
            near,far=groups[key][ni],groups[key][fi]
            assert pair['near_distance']==near['distance_to_goal'] and pair['far_distance']==far['distance_to_goal']
            assert pair['near_reward']==near['total_reward'] and pair['far_reward']==far['total_reward']
            assert pair['far_ndtw_reward']==far['ndtw_reward']
    assert recorded==all_pairs
    assert {run:dict(c) for run,c in counts.items()}==report['summaries']
    assert len(groups)==2048
    out={'schema':'within_group_reward_order_independent_compact_recount_v1','status':'PASS','training_records':8192,'groups':2048,'rank_conflict_groups':len(case_keys),'rank_conflict_pairs':len(all_pairs),'summaries':{run:dict(c) for run,c in counts.items()},'source_compact_sha256':sha(a.source),'report_sha256':sha(a.report),'verifier_sha256':sha(Path(__file__)),'scope':'Independent set/pair/count arithmetic on the existing compact source; not second raw parsing, model advantage reconstruction, physical replay, semantic accuracy or navigation benefit.'}
    with a.output.open('x') as f:f.write(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='summaries'}))


if __name__=='__main__':
    main()
