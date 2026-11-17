import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).parents[2]))
import cluster.transfer.dispatch as D
import cluster.transfer.scan as S
import test_nightly_transfer_dispatch as fixtures


def nights():
    return [{'start':f'2026-11-{day}T00:00:00+08:00','end':f'2026-11-{day}T08:00:00+08:00'} for day in (13,14,15)]


class DailyTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.DispatchTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.old = self.f.config
        self.old['sources']['sensecore']['expires_at'] = D.timestamp('2026-11-15T14:45:00+08:00')
        # Fixture was initially created before adding expiry. This is setup, not migration.
        self.f.scheduler.state['config_sha256'] = D.sha(D.canonical(self.old))
        self.f.scheduler.save()
        self.new = {k:v for k,v in copy.deepcopy(self.old).items() if k not in ('window_start','window_end')}
        self.new['windows'] = nights()
        self.old_path,self.new_path = self.f.root/'old.json',self.f.root/'new.json'
        self.old_path.write_text(json.dumps(self.old))
        self.new_path.write_text(json.dumps(self.new))

    def migrate(self):
        return D.migrate_config(self.old_path,self.new_path,D.file_sha(self.old_path),D.file_sha(self.new_path))

    def test_strict_daily_windows_and_expiry(self):
        windows = D.parse_windows(self.new)
        for start,end in windows:
            self.assertIsNotNone(D.active_window(windows,start))
            self.assertIsNone(D.active_window(windows,end))
            self.assertIsNone(D.active_window(windows,start-1))
        for mutate in ('hours','expiry','gap','empty','mixed'):
            cfg = copy.deepcopy(self.new)
            if mutate=='hours': cfg['windows'][0]['end']='2026-11-13T09:00:00+08:00'
            if mutate=='expiry': cfg['sources']['sensecore']['expires_at']=windows[-1][1]-1
            if mutate=='gap': cfg['windows'].pop(1)
            if mutate=='empty': cfg['windows']=[]
            if mutate=='mixed': cfg['window_start']=self.old['window_start']
            with self.assertRaises(ValueError): D.parse_windows(cfg)

    def test_migration_preserves_state_receipts_and_partial_and_is_idempotent(self):
        pid = self.f.add(0)
        self.f.scheduler.discover(self.f.s,self.f.remote)
        model = self.f.scheduler.state['models'][pid]
        model.update(phase='ERROR',transfer_started_at=self.f.s+1,retry_at=self.f.s+10)
        self.f.scheduler.save()
        folder = self.f.scheduler.root/'models'/pid
        (folder/'actor.pt.part').write_bytes(b'precious partial')
        (folder/'evaluation-intent.json').write_text('{"pending":"do not repost"}')
        before = {p.name:p.read_bytes() for p in folder.iterdir() if p.is_file()}
        expected = copy.deepcopy(self.f.scheduler.state['models'])
        receipt = self.migrate()
        scheduler = D.Scheduler(self.new)
        self.assertEqual(scheduler.state['models'],expected)
        self.assertEqual({p.name:p.read_bytes() for p in folder.iterdir() if p.is_file()},before)
        self.assertFalse(receipt['already_applied'])
        scheduler.state['models'][pid]['phase']='EVALUATION_QUEUED'
        scheduler.save()
        self.assertTrue(self.migrate()['already_applied'])
        self.assertEqual(D.Scheduler(self.new).state['models'][pid]['phase'],'EVALUATION_QUEUED')

    def test_locked_or_unaudited_migration_does_not_write(self):
        state = self.f.scheduler.state_path
        before = state.read_bytes()
        with (self.f.scheduler.root/'scheduler.lock').open('a+') as lock:
            D.fcntl.flock(lock,D.fcntl.LOCK_EX|D.fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError): self.migrate()
        self.assertEqual(state.read_bytes(),before)
        with self.assertRaises(D.IdentityError): D.migrate_config(self.old_path,self.new_path,'0'*64,D.file_sha(self.new_path))
        self.new['sources']['sensecore']['ssh_alias']='other'
        self.new_path.write_text(json.dumps(self.new))
        with self.assertRaisesRegex(D.IdentityError,'only change schedule'): self.migrate()
        self.assertEqual(state.read_bytes(),before)

    def test_migration_interruption_after_state_save_recovers_without_reset(self):
        original = D.immutable
        def interrupt(path,value):
            if path.name=='APPLIED.json': raise OSError('simulated crash after atomic state')
            return original(path,value)
        with patch.object(D,'immutable',interrupt):
            with self.assertRaises(OSError): self.migrate()
        state = json.loads(self.f.scheduler.state_path.read_text())
        self.assertEqual(state['config_sha256'],D.sha(D.canonical(self.new)))
        self.assertTrue(self.migrate()['already_applied'])
        self.assertEqual(json.loads(self.f.scheduler.state_path.read_text()),state)

    def test_cross_night_pending_priority_retry_and_dedup(self):
        first = self.f.add(0)
        self.f.scheduler.discover(self.f.s,self.f.remote)
        self.migrate()
        scheduler = D.Scheduler(self.new)
        day1,day2,day3 = scheduler.windows
        self.assertEqual(scheduler.next_model(day1[0])['policy_id'],first)
        self.assertIsNone(scheduler.next_model(day1[1]))
        self.assertIsNone(scheduler.next_model(day2[0]-1))
        scheduler.close_report()
        scheduler.open_window(day2[0])
        self.assertFalse(scheduler.state['window_closed'])
        self.assertEqual(scheduler.state['unprocessed'],[])
        self.assertTrue(scheduler.state['close_report_history'][0]['unprocessed'])
        late = self.f.add(1)
        scheduler.discover(day2[0],self.f.remote)
        self.assertEqual(scheduler.next_model(day2[0])['policy_id'],late)
        scheduler.state['models'][late].update(phase='EVALUATION_QUEUED')
        scheduler.save()
        scheduler = D.Scheduler(self.new)
        self.assertEqual(scheduler.next_model(day3[0])['policy_id'],first)
        scheduler.state['models'][first].update(transfer_started_at=day3[1]-1,started_at=day3[1]-1,phase='ERROR',retry_at=day3[1]+60)
        self.assertTrue(scheduler.active_retry_pending(day3[1]))
        self.assertEqual(scheduler.next_model(day3[1]+60)['policy_id'],first)
        self.assertFalse(scheduler.active_retry_pending(self.new['sources']['sensecore']['expires_at']))

    def test_real_run_loop_stays_between_three_nights(self):
        first = self.f.add(0)
        self.migrate()
        windows = D.parse_windows(self.new)
        moments = [windows[0][0],windows[0][1],windows[1][0]-1,windows[1][0],windows[1][1],windows[2][0],windows[2][1]]
        current = [moments[0]]
        def sleep(seconds):
            if len(moments)>1: moments.pop(0)
            else: self.fail('Loop failed to exit after finite final window')
            current[0]=moments[0]
        def remote(argv,**kwargs):
            if current[0]>=windows[1][0] and len(self.f.catalog)==1: self.f.add(1)
            args = D.shlex.split(argv[-1])
            return type('Reply',(),{'returncode':0,'stderr':b'','stdout':D.canonical(self.f.remote('sensecore',args))})()
        started = []
        class ImmediatePool:
            def __init__(self,**kwargs): pass
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def submit(self,func,scheduler,model,env):
                started.append(model['policy_id'])
                model['phase']='EVALUATION_QUEUED'
                scheduler.save()
                return type('Done',(),{'done':lambda self:True,'result':lambda self:None})()
        self.new['sources']['sensecore'].update(ssh_config='unused',ssh_alias='source')
        # New fixture config includes transport fields before it is first run.
        state = json.loads(self.f.scheduler.state_path.read_text())
        state['config_sha256']=D.sha(D.canonical(self.new))
        D.atomic(self.f.scheduler.state_path,state)
        with patch.object(D.time,'time',lambda:current[0]),patch.object(D.time,'sleep',sleep), \
             patch.object(D.subprocess,'run',remote),patch.object(D,'ThreadPoolExecutor',ImmediatePool):
            final = D.run(self.new)
        self.assertEqual(final['closed_windows'],[end for _,end in windows])
        self.assertEqual(final['models'][first]['phase'],'EVALUATION_QUEUED')
        self.assertEqual(len(started),2)
        self.assertEqual(len(set(started)),2)
        self.assertEqual(len(final['close_report_history']),2)
        self.assertIsNone(final['next_window_start'])

    def test_source_scanner_continues_past_first_night_and_stops_finitely(self):
        root = self.f.root/'scan'
        source,contract,schedule = self.f.root/'source.py',self.f.root/'contract.json',self.f.root/'schedule.json'
        source.write_text('pinned source')
        windows = D.parse_windows(self.new)
        contract.write_text(json.dumps({'discovery_end':windows[-1][1],'expires_at':self.new['sources']['sensecore']['expires_at']}))
        schedule.write_text(json.dumps({'windows':nights()}))
        args = type('Args',(),dict(source=source,contract=contract,root=root,source_sha256=S.digest(source),
            contract_sha256=S.digest(contract),schedule=schedule,schedule_sha256=S.digest(schedule)))()
        now = [windows[0][0]]
        scans = []
        def sleep(seconds): now[0]+=seconds
        def invoke(*a,**k):
            scans.append(now[0])
            return type('Reply',(),{'returncode':0,'stdout':'{"ready":[]}','stderr':''})()
        final = S.run(args,lambda:now[0],sleep,invoke)
        self.assertTrue(any(t>windows[0][1] for t in scans))
        self.assertTrue(all(end-300 in scans for _,end in windows))
        self.assertEqual(scans[-1],windows[-1][1]-300)
        self.assertIsNone(final['next_scan'])
        contract.write_text(json.dumps({'discovery_end':windows[0][1],'expires_at':windows[0][1]}))
        args.contract_sha256=S.digest(contract)
        with self.assertRaisesRegex(ValueError,'exceeds'): S.run(args)


if __name__ == '__main__': unittest.main(verbosity=2)
