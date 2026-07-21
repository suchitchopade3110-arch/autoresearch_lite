from typing import List, Dict, Any, Tuple
import math

def dominates(scores_a: Dict[str, float], scores_b: Dict[str, float], maximize: Dict[str, bool]) -> bool:
    at_least_as_good_in_all = True
    strictly_better_in_one = False

    for metric in maximize.keys():
        val_a = scores_a.get(metric, 0.0)
        val_b = scores_b.get(metric, 0.0)

        if maximize[metric]:
            if val_a < val_b:
                at_least_as_good_in_all = False
                break
            if val_a > val_b:
                strictly_better_in_one = True
        else:
            if val_a > val_b:
                at_least_as_good_in_all = False
                break
            if val_a < val_b:
                strictly_better_in_one = True

    return at_least_as_good_in_all and strictly_better_in_one

def calculate_pareto_fronts(candidates: List[Dict[str, Any]], maximize: Dict[str, bool]) -> List[int]:
    n = len(candidates)
    ranks = [0] * n

    for i in range(n):
        domination_count = 0
        for j in range(n):
            if i == j:
                continue
            if dominates(candidates[j]['metrics'], candidates[i]['metrics'], maximize):
                domination_count += 1
        ranks[i] = domination_count

    return ranks

def score_candidates(candidates: List[Dict[str, Any]], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    strategy = config.get('scoring_strategy', 'pareto')
    watts_constant = config.get('energy_watts_constant', 10.0)

    for c in candidates:
        m = c.get('metrics', {})
        if 'energy_estimate' not in m and 'execution_time' in m:
            m['energy_estimate'] = m['execution_time'] * watts_constant

        m.setdefault('validation_accuracy', 0.0)
        m.setdefault('execution_speed', 0.0)
        m.setdefault('memory_usage', 1024.0)
        m.setdefault('energy_estimate', 100.0)

        c['metrics'] = m

    maximize = {
        'validation_accuracy': True,
        'execution_speed': True,
        'memory_usage': False,
        'energy_estimate': False
    }

    if strategy == 'weighted':
        weights = config.get('weights', {
            'validation_accuracy': 1.0,
            'execution_speed': 0.2,
            'memory_usage': 0.1,
            'energy_estimate': 0.1
        })

        for c in candidates:
            m = c['metrics']
            score = (
                m['validation_accuracy'] * weights.get('validation_accuracy', 1.0) +
                m['execution_speed'] * weights.get('execution_speed', 1.0) -
                (math.log(max(m['memory_usage'], 1.0)) * weights.get('memory_usage', 1.0)) -
                (math.log(max(m['energy_estimate'], 1.0)) * weights.get('energy_estimate', 1.0))
            )
            c['composite_score'] = score

    elif strategy == 'pareto':
        ranks = calculate_pareto_fronts(candidates, maximize)
        for i, c in enumerate(candidates):
            c['composite_score'] = -ranks[i]
            c['pareto_rank'] = ranks[i]

    else:
        raise ValueError(f"Unknown scoring strategy: {strategy}")

    return candidates