"""Build a local reproduction bundle from explicit prepared inputs and membership."""
import argparse
import csv
import json
from pathlib import Path



def read(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def write(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'cell-mean', 'original-response', 'cell-mean-protocol', 'original-response-protocol', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        parser.error('Output must be an empty directory or a new path.')
    manifest = read(args.manifest)
    used = set()
    for panel in manifest:
        for name in ('train_source_row_ids_json', 'test_source_row_ids_json'):
            used.update(json.loads(panel[name]))
    variants = {}
    for label in ('cell_mean', 'original_response'):
        archive = read(getattr(args, label))
        rows = [r for r in archive if int(r['source_table_row_id']) in used]
        assert len(rows) == len(used) == 595
        assert {r['pollutant'] for r in rows} == {'IBU', 'Sr (II)'}
        assert len({r['source_study_id'] for r in rows}) == 6
        raw_ids = [i for r in rows for i in json.loads(r['raw_record_ids_json'])]
        assert len(raw_ids) == len(set(raw_ids)) == 1704
        config = json.loads(getattr(args, label+'_protocol').read_text())
        for r in rows:
            assert not any('/public/home/' in str(v) for v in r.values())
        variants[label] = (rows, config)
    left, right = (variants[k][0] for k in ('cell_mean', 'original_response'))
    for a, b in zip(left, right):
        assert all(b[k] == v for k, v in a.items())
    out.mkdir(parents=True, exist_ok=True)
    write(out/'manifest.csv', manifest)
    for label, (rows, config) in variants.items():
        write(out/f'{label}.csv', rows)
        (out/f'{label}_protocol.json').write_text(json.dumps(config, indent=2)+'\n')
    report = dict(status='local_reproduction_only_distribution_terms_not_verified',
                  cells=595, linked_released_records=1704, source_folds=6,
                  unused_cbz_cells_excluded=90, identical_cell_values=True,
                  limitations=['Not independent experimental replicate counts',
                               'No publication or source-license grant implied'])
    (out/'bundle_status.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
