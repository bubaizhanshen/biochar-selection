"""Compare aggregate numerical ranges with the reference environment result."""
import argparse
from pathlib import Path
import pandas as pd


def check(summary, reference):
    keys=['policy','pollutant','model_count']
    got=pd.read_csv(summary).set_index(keys).sort_index()
    expected=pd.read_csv(reference).set_index(keys).sort_index()
    pd.testing.assert_frame_equal(got,expected,rtol=1e-10,atol=1e-10)
    print('Eight task/model-set aggregate ranges match the numerical reference.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary',type=Path,required=True)
    parser.add_argument('--reference',type=Path,
        default=Path(__file__).resolve().parents[1]/'tests/fixtures/numerical_reference.csv')
    args=parser.parse_args()
    check(args.summary,args.reference)
