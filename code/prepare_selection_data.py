"""Reconstruct both six-source selection inputs from the released SI workbooks."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hmi-workbook', type=Path, required=True)
    parser.add_argument('--ec-workbook', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    for path in (args.hmi_workbook, args.ec_workbook):
        if not path.is_file():
            parser.error(f'Workbook not found: {path}')
    out = args.out.resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        parser.error('Output must be an empty directory or a new path.')
    out.mkdir(mode=0o700, parents=True, exist_ok=True)
    cache = out / 'cache'
    cache.mkdir(mode=0o700)
    env = os.environ.copy()
    for key in ('TMPDIR', 'TMP', 'TEMP', 'XDG_CACHE_HOME'):
        env[key] = str(cache)
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[key] = '1'
    scripts = ROOT / 'code/preparation'
    mappings = ROOT / 'data/selection_annotations'
    source, units, cells, split, raw, bundle = [out / x for x in
        ('source_copies', 'units', 'cells', 'split', 'raw_responses', 'bundle')]

    def run(name, *arguments):
        subprocess.run([sys.executable, str(scripts / name), *map(str, arguments)],
                       env=env, check=True)

    run('build_source_analysis_copies.py',
        '--hmi-workbook', args.hmi_workbook.resolve(),
        '--ec-workbook', args.ec_workbook.resolve(), '--mappings', mappings, '--out', source)
    run('harmonize_competitive_ibu_units.py', '--use-reviewed-source-units',
        '--source-copy', source / 'shin2021_competitive_analysis_copy_candidate.csv',
        '--series-mapping', mappings / 'shin2021_competitive_series_mapping_review.csv', '--out', units)
    run('build_public_source_selection_copy.py', '--source-copies', source,
        '--competitive-unit-copy', units / 'shin2021_competitive_ibu_unit_harmonized.csv', '--out', cells)
    run('build_public_selection_manifest.py', '--cells', cells / 'public_source_cells.csv',
        '--base-protocol', ROOT / 'config/selection.json', '--out', split)
    run('build_raw_response_sensitivity.py', '--cells', split / 'public_pilot_input.csv',
        '--source-rows', cells / 'public_source_rows.csv',
        '--base-protocol', split / 'public_pilot_protocol.json', '--out', raw)
    run('build_selection_reproduction_bundle.py', '--manifest', split / 'public_pilot_manifest.csv',
        '--cell-mean', split / 'public_pilot_input.csv',
        '--original-response', raw / 'public_raw_response_input.csv',
        '--cell-mean-protocol', split / 'public_pilot_protocol.json',
        '--original-response-protocol', raw / 'public_raw_response_protocol.json', '--out', bundle)
    print(f'Prepared inputs: {bundle}')
    print('Intermediate files contain source-derived data; do not publish them without checking source terms.')


if __name__ == '__main__':
    main()
