from __future__ import division
import numpy as np
from scipy.signal import firwin, filtfilt
try:
    from itertools import izip as zip
except ImportError:
    pass

# filters
filter = False #  TODO
def streamify(func, *args, **kwargs):
    return lambda stream: (func(x) for x in stream)
abs = streamify(np.abs)


def streamify_reducer(func, *args, **kwargs):
    """ takes a reducing operations, such as numpy.mean() or numpy.median()
    and applies it at each time step to all the streams.

    Thus ten samples from four streams would return ten means.

    For example
    >>> a = ((1,2,3), (2,4,9))
    >>> s_sum = streamify_reducer(sum)
    >>> list(s_sum(a))
    [3, 6, 12]
    """
    return lambda *streams: (func(x, *args, **kwargs) for x in zip(*streams))
    return a

sum = streamify(np.sum)
mean = streamify(np.mean)
median = streamify(np.median)

def select(stream, columns=()):
    for row in stream:
        if columns:
            yield row[columns]
        else:
            yield row

def merge(*streams):
    return (np.hstack(x) for x in zip(*streams))


def chunk(func, length=90000, overlap=20000):
    """some functions, such as filters, require chunked input
    chunks have three sections:
    | overlap/2 | length - overlap | overlap/2 |

    all three sections are sent to a function, but only the center of the chunk is returned as output,
    except for the first and last chunks, which include the left and right overlaps respectively."""
    assert length > overlap
    def new_func(stream, *args, **kwargs):
        first_chunk = True
        last_chunk = False
        buffer = []
        while not last_chunk:
            while (len(buffer) < length) and not last_chunk:
                try:
                    buffer.append(next(stream))
                except StopIteration:
                    last_chunk = True
            chunk = func(np.array(buffer), *args, **kwargs)
            print("chunk: ", chunk)
            ichunk = iter(chunk)
            # iterate over first overlap, yielding if it's the first chunk
            for i in range(overlap // 2):
                if first_chunk:
                    print("right")
                    yield next(ichunk)
                else:
                    next(ichunk)
            first_chunk = False
            # yield the center non-overlapping part of the chunk
            for i in range(length - overlap):
                print("center")
                yield next(ichunk)
            # yield last overlap if there's no more data in the stream
            if last_chunk:
                for x in ichunk:
                    print("left")
                    yield x
            buffer = buffer[-overlap:]
    return new_func

@chunk
def dummy_filter(x, n=0):
    return x+n


def _filtfilt(x, b, a, *args, **kwargs):
    """helper function to reorder arguments to filtfilt,
    so the data is the first argument"""
    print("a is ", a)
    print("b is", b)
    print("x is", x)
    return filtfilt(b, a, x, *args, **kwargs)
# eeg filter: https://bitbucket.org/sccn_eeglab/eeglab/src/a2983535293a3bd3605e96a806182d6d6a4a2594/functions/sigprocfunc/eegfilt.m?at=master&fileviewer=file-view-default
#@chunk
def fireegfilt(stream, sampling_rate, lowcut=None, highcut=None, filter_order=5, **kwargs):
    assert lowcut is None or 0 < lowcut < sampling_rate/2
    assert highcut is None or 0 < highcut < sampling_rate/2
    nyq = sampling_rate / 2
    if lowcut is not None and highcut is not None:
        a = firwin(filter_order, [lowcut/nyq, highcut/nyq], pass_zero=False)
    elif lowcut is not None:  # high pass filter
        a = firwin(filter_order, lowcut/nyq, pass_zero=False)
    elif highcut is not None: # low pass filter
        a = firwin(filter_order, highcut/nyq)
    else:
        raise Exception("lowcut or highcut must be specified")
    return _filtfilt(stream, np.array([1]), a)

def iireegfilt(stream, sampling_rate, lowcut=None, highcut=None, filter_order=5, **kwargs):
    pass
