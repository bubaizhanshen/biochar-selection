import contextlib
import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_selection_data as preparation
import pandas as pd
from preparation import build_source_analysis_copies as source_copies


class SelectionPreparationTest(unittest.TestCase):
    def test_public_hmi_headers_preserve_values_and_row_order(self):
        raw = pd.DataFrame({'inorganics': ['Sr(II)', 'Cd(II)'],
                            'g/L': [1.0, 2.0], '(O+N/C)': [0.1, 0.2]})
        with patch.object(source_copies.pd, 'read_excel', return_value=raw):
            frame = source_copies._raw_frame('Dataset II')
        self.assertEqual(frame.columns.tolist(),
                         ['source_row_id', 'Metal type', 'Dosage(g/L)', '(O+N)/C'])
        self.assertEqual(frame['source_row_id'].tolist(), [0, 1])
        self.assertEqual(frame['Dosage(g/L)'].tolist(), [1.0, 2.0])

    def test_public_ec_header_whitespace_is_removed(self):
        raw = pd.DataFrame({'Pyrolysis temperature ': [500]})
        with patch.object(source_copies.pd, 'read_excel', return_value=raw):
            frame = source_copies._raw_frame('Dataset III')
        self.assertEqual(frame['Pyrolysis temperature'].tolist(), [500])

    def test_ambiguous_header_aliases_fail(self):
        raw = pd.DataFrame({'inorganics': ['Sr(II)'], 'Metal type': ['Cd(II)']})
        with patch.object(source_copies.pd, 'read_excel', return_value=raw):
            with self.assertRaisesRegex(ValueError, 'ambiguous duplicate'):
                source_copies._raw_frame('Dataset II')

    def setUp(self):
        cache = preparation.ROOT / 'work/test_cache'
        cache.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=cache)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.hmi, self.ec = self.root / 'HMI.xlsx', self.root / 'EC.xlsx'
        self.hmi.touch()
        self.ec.touch()
        self.out = self.root / 'prepared'

    def call(self):
        argv = ['prepare', '--hmi-workbook', str(self.hmi),
                '--ec-workbook', str(self.ec), '--out', str(self.out)]
        with patch('sys.argv', argv), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            preparation.main()

    def test_missing_workbook_fails_before_output_creation(self):
        self.ec.unlink()
        with self.assertRaises(SystemExit):
            self.call()
        self.assertFalse(self.out.exists())

    def test_nonempty_output_is_not_overwritten(self):
        self.out.mkdir()
        sentinel = self.out / 'existing'
        sentinel.touch()
        with self.assertRaises(SystemExit):
            self.call()
        self.assertTrue(sentinel.exists())

    def test_all_six_stages_use_explicit_paths(self):
        with patch.object(preparation.subprocess, 'run') as run:
            self.call()
        self.assertEqual(run.call_count, 6)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertTrue(commands[0][1].endswith('build_source_analysis_copies.py'))
        self.assertIn('--use-reviewed-source-units', commands[1])
        self.assertTrue(commands[-1][1].endswith('build_selection_reproduction_bundle.py'))
        for call in run.call_args_list:
            self.assertTrue(call.kwargs['check'])
            self.assertEqual(call.kwargs['env']['TMPDIR'], str(self.out / 'cache'))
            self.assertNotIn('estl/preparation/audit', ' '.join(call.args[0]))

    def test_annotations_have_unique_locators_and_no_measurement_columns(self):
        paths = list((preparation.ROOT / 'data/selection_annotations').glob('*.csv'))
        self.assertEqual(len(paths), 6)
        forbidden = {'Capacity', 'qe', 'Ci', 'Cf', 'Initial concentration',
                     'Final concentration', 'Surface area', 'C', 'H', 'N', 'O', 'Ash'}
        for path in paths:
            with path.open(newline='') as handle:
                reader = csv.DictReader(handle)
                self.assertFalse(forbidden.intersection(reader.fieldnames))
                ids = [int(row['source_row_id']) for row in reader]
            self.assertTrue(ids)
            self.assertEqual(len(ids), len(set(ids)))


if __name__ == '__main__':
    unittest.main()
