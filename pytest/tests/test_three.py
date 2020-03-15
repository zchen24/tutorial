#!/usr/bin/env python3

"""
Exampe from Python Testing with pytest by Brian Okken
"""

from collections import namedtuple

Task = namedtuple('Task', ['summary', 'owner', 'done', 'id'])
Task.__new__.__defaults__ = (None, None, False, None)


def test_defaults():
    """Using no parameters should invoke defaults."""
    t1 = Task()
    t2 = Task(None, None, False, None)
    assert  t1 == t2

def test_member_access():
    """Check .field functionality of namedtuple."""
    t = Task('buy milk', 'brian')
    assert t.summary == 'buy milk'
    assert t.owner == 'brian'
    assert (t.done, t.id) == (False, None)

def test_asdict():
    """asdict() should return a dictionary."""
    t_task = Task('do something', 'chen', True, 21)
    t_dict = t_task._asdict()
    expected = {'summary': 'do something',
                'owner': 'chen',
                'done': True,
                'id': 21}
    assert t_dict == expected

def test_replace():
    """replace() should change passed in fields"""
    t_before = Task('finish book', 'chen', False)
    t_after = t_before._replace(id=10, done=True)
    t_expected = Task('finish book', 'chen', True, 10)
    assert t_after == t_expected
