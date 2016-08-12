from dspflow import Stream, ArfStreamer
import numpy as np
import os

def get_simple_stream():
    return Stream(np.arange(3*i, 3*i+3) for i in range(10))

def rowlen(arr):
    return arr.shape[0]


def test_ensureNumElems():
    stm = get_simple_stream()
    stm._ensureNumElems(10)
    assert rowlen(stm.head) >= 10

def test_peek_doesnt_modify():
    stm = get_simple_stream()
    a = stm.peek(3)
    b = stm.peek(3)
    assert np.array_equal(a,b)

def test_peek_returns_fewer_when_stream_too_short():
    stm = get_simple_stream()
    a = stm.peek(45)
    assert rowlen(a) == 30

def test_readElements_basic():
    stm = get_simple_stream()
    a = stm.read(4)
    assert np.array_equal(a, np.arange(0, 4))
    b = stm.read(3)
    assert np.array_equal(b, np.arange(4,7))
    c = stm.read(100)
    assert rowlen(a)+rowlen(b)+rowlen(c) == 30

def test_map():
    stm = get_simple_stream()
    stm.peek(3) # To see if head is properly preserved
    stm = stm.map(lambda x: 2*x)
    a = stm.peek(3)
    assert np.array_equal(a, 2*np.arange(0,3))
    b = stm.read(5)
    assert np.array_equal(b, 2*np.arange(0,5))


def test_chunked():
    stm = get_simple_stream()
    chunked = stm.chunked(3,1)
    vals = chunked.read(2)
    assert np.array_equal(vals[0], np.array([0,1,2]))
    assert np.array_equal(vals[1], np.array([2,3,4]))


def test_merge():
    stm1 = get_simple_stream()
    stm2 = get_simple_stream()
    merged = Stream.merge(stm1, stm2)
    a = merged.read(3)
    assert np.array_equal(a, np.array([[0,0],[1,1],[2,2]]))

def test_merge_with_2d():
    stm1 = get_simple_stream()
    stm2 = Stream.merge(get_simple_stream(), get_simple_stream(), chunk_size=3)
    merged = Stream.merge(stm1, stm2, chunk_size=5)
    a = merged.read(100)[-1]
    assert np.array_equal(a, np.array([29,29,29]))

def test_ArfStreamer():
    filename = "test.arf"
    path = "/path/to/be/tested"
    stm1 = get_simple_stream()
    ArfStreamer.save(stm1, filename, path, chunk_size=10, sampling_rate=40000)
    with ArfStreamer(filename) as astm:
        stm2 = astm.stream_channel(path, chunk_size=15)
        read2 = stm2.read(1000)
    assert np.array_equal(read2, np.arange(0,30))

    channels = ['entry/channel'+str(i) for i in range(3)]
    for i in range(3):
        stm = get_simple_stream()
        ArfStreamer.save(stm.map(lambda chunk, i=i: chunk * i),
            filename, channels[i], chunk_size=10, sampling_rate=40000)
    
    with ArfStreamer(filename) as astm:
        read_stm = astm.stream_channels(channels, chunk_size=15)
        read = read_stm.read(1000)
    print(read)
    assert np.array_equal(read[:,i], np.arange(0,30)*i)
    os.remove('test.arf')


