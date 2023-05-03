#!/usr/bin/env python
from dnfunc import NumFramesError


def test_numframeserror():
    assert issubclass(NumFramesError, Exception)
