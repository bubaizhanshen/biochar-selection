"""Add the declared nine-model set to an existing fixed-parameter protocol."""
import argparse
import json
from pathlib import Path
from model_registry import configured_models


def expand(base, extension):
    if 'inner_parameter_candidates' in base:
        raise ValueError('Expansion requires a fixed-parameter base protocol')
    result = dict(base)
    result.update(extension)
    configured_models(result)
    result['model_selection'] = 'Fixed parameters; family selected by inner sources only; no outer-response tuning.'
    result['model_expansion'] = {
        'scope': 'Retrospective model-family extension on previously examined data; not new external validation',
        'rationale': 'Add commonly used nearest-neighbor, randomized-tree and boosting regressors; CEJ 2023, 466, 143073',
        'preprocessing': 'Identical training-only scaling, target standardization and one-hot encoding for all families; no native categorical mode',
        'search_budget': 'One fixed configuration per family, no early stopping or tuning against outer responses',
        'tie_order': 'Declared model order; surface_area then random precede models when rules compete',
        'comparison': 'Same source holdouts, conditions, candidate sets, observed responses and aggregation as three-model comparison'
    }
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--extension',type=Path,default=Path(__file__).resolve().parents[1]/'config/model_expansion.json')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(expand(json.loads(args.base.read_text()),json.loads(args.extension.read_text())),indent=2)+'\n')
