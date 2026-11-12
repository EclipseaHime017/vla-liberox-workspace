import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

spec = importlib.util.spec_from_file_location('scan', Path(__file__).parents[1] / 'scripts/nightly_transfer_scan.py')
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


class ScanTests(unittest.TestCase):
    def test_schedule_final_scan_and_exception_survive(self):
        with tempfile.TemporaryDirectory(prefix='r21-scan-') as tmp:
            root = Path(tmp)
            source, contract = root/'source.py', root/'contract.json'
            source.write_text('pinned')
            contract.write_text(json.dumps({'discovery_end': 4300}))
            args = argparse.Namespace(source=source, contract=contract, root=root/'ops',
                source_sha256=S.digest(source), contract_sha256=S.digest(contract))
            now = [0]
            def sleep(seconds): now[0] += seconds
            call = Mock(side_effect=[OSError('offline'), *[argparse.Namespace(returncode=0, stdout='{"ready":[]}', stderr='')]*3])
            state = S.run(args, lambda: now[0], sleep, call)
            self.assertEqual(call.call_count, 4)
            self.assertEqual(state['last_scan'], 4000)
            self.assertIsNone(state['next_scan'])
            lines = (args.root/'scanner-events.jsonl').read_text().splitlines()
            self.assertEqual(json.loads(lines[0])['error'], 'offline')
            self.assertEqual([json.loads(l)['last_scan'] for l in lines], [0,1800,3600,4000])
            contract.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'SHA mismatch'): S.run(args)

    def test_due_boundaries(self):
        self.assertEqual(S.next_scan(100, None, 4000), 100)
        self.assertEqual(S.next_scan(5000, 3600, 4000), 4000)
        self.assertIsNone(S.next_scan(4100, 4000, 4000))


if __name__ == '__main__': unittest.main(verbosity=2)
