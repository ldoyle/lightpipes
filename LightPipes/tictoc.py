# -*- coding: utf-8 -*-
"""
Imitation of matlabs tic toc functions. Uses a stack and popping so that nested
timings can be done. Source:
https://stackoverflow.com/questions/5849800/tic-toc-functions-analog-in-python
Answer by user Stephan

Note: probably not thread safe

Created on Sun Jul 23 15:13:12 2017

@author: Lenny
"""

from time import time
_tstart_stack = []

def tic():
    """Start a new measurement by adding current time() to timing stack"""
    _tstart_stack.append(time())

def toc():
    """End the most recent measurement and return elapsed time in [s]"""
    return time() - _tstart_stack.pop()

def strtoc(fmt='', prefix=''):
    """End the most recent measurement and return elapsed time as string,
    given the format string"""
    if not fmt:
        fmt = "%.4ss"
    return prefix + fmt % toc()

def printtoc(fmt='', prefix=''):
    """End the most recent measurement and print the time to console given
    the format string"""
    print(strtoc(fmt, prefix))

def timeit(method):
    """Use as decorator for any function
    @timeit
    def myfunc(args):
        #do stuff
        return
    """
    """Inspired from
    https://medium.com/pythonhive/python-decorator-to-measure-the-execution
        -time-of-methods-fa04cb6bb36d
    """
    def timed(*args, **kw):
        tic()
        result = method(*args, **kw)
        printtoc(prefix='Time for '+method.__name__+': ')
        return result
    return timed

