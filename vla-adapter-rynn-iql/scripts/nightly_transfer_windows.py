"""Explicit finite R22 night windows; never extends a source capability."""
from datetime import datetime, timedelta


def parse_windows(config):
    if 'windows' not in config:
        start, end = (datetime.fromisoformat(config[k]) for k in ('window_start','window_end'))
        if start.utcoffset() is None or end.utcoffset() is None or start >= end:
            raise ValueError('Invalid timezone-aware window')
        return [(start.timestamp(),end.timestamp())]
    if 'window_start' in config or 'window_end' in config:
        raise ValueError('Do not mix old and new window definitions')
    rows = config['windows']
    if not isinstance(rows,list) or not 1 <= len(rows) <= 31:
        raise ValueError('Require 1 to 31 explicitly approved nights')
    result = []
    for row in rows:
        if set(row) != {'start','end'}: raise ValueError('Invalid window keys')
        start, end = (datetime.fromisoformat(row[k]) for k in ('start','end'))
        if (start.utcoffset() != timedelta(hours=8) or end.utcoffset() != timedelta(hours=8)
                or (start.hour,start.minute,start.second,start.microsecond) != (0,0,0,0)
                or (end.hour,end.minute,end.second,end.microsecond) != (8,0,0,0)
                or start.date() != end.date()):
            raise ValueError('R22 windows must be 00:00–08:00 Asia/Singapore')
        if result and start.timestamp() != result[-1][0]+86400:
            raise ValueError('Night list must be chronological consecutive dates')
        result.append((start.timestamp(),end.timestamp()))
    for source in config.get('sources',{}).values():
        if source.get('expires_at',0) < result[-1][1]:
            raise ValueError('Window exceeds pinned source expiry')
    return result


def active_window(windows, now):
    return next(((start,end) for start,end in windows if start <= now < end),None)
