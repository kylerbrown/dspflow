import numpy as np
import copy
import arf


class ArfStreamer():
    def open(filename, path, chunk_size=500000):
            def gen(filename, path, chunk_size):
                with arf.open_file(filename) as file:
                    path = path.split("/")
                    if path[0]=="":
                        path = path[1:]
                    dataset = file
                    for p in path:
                        dataset = dataset[p]
                    length = len(dataset)
                    offset = 0
                    while offset < length:
                        if length - offset > chunk_size:
                            chunk = chunk_size
                        else:
                            chunk = length - offset
                        print("reading new batch")
                        buffer = dataset[offset:offset+chunk]
                        yield buffer
                        offset += chunk
            return Stream(gen(filename, path, chunk_size))

    def save(stream, filename, path, chunk_size=1000000):
        with arf.open_file(filename, 'w') as file:
            path = path.split("/")
            dst_name = path[-1]
            grp_path = "/".join(path[:-1])
            grp = file.create_group(grp_path)
            # TODO check if already exists
            
            #Get first batch of data
            # data = stream.readElements(chunk_size)
            data = stream.read(chunk_size)
            dst = arf.create_dataset(grp, dst_name, data,
                maxshape=(None,), sampling_rate=40000)
            # dst = grp.create_dataset(dst_name, data=data, maxshape=(None,))
            while True:
                data = stream.read(chunk_size)
                if len(data) == 0:
                    break
                arf.append_data(dst, data)

class Stream():
    def __init__(self, gen, head = None):
        self.gen = gen
        # Head stores the chunk we get from the generator,
        # so that we can peek or read however many elements we want.
        if head != None:
            self.head = head
        else:
            # We take a chunk immediately, to learn the shape.
            self.head = next(self.gen)

        self.element_shape = self.head[0].shape

    def _advanceChunk(self):
        chunk = next(self.gen)[:] #to copy, in case
        if self.head.ndim == 1:
            self.head = np.hstack([self.head, chunk])
        elif self.head.ndim == 2 or self.head.ndim == 3:
            #ndim == 3 in case of 2d chunked data
            self.head = np.vstack([self.head, chunk])
        else:
            raise

    def _ensureNumElems(self, numElems):
        while self.head.shape[0] < numElems:
            try:
                self._advanceChunk()
            except StopIteration:
                return

    def get_iter(self, numPerIter=1):
        while True:
            els = self.read(numPerIter)[:]
            if els.shape[0] == 0:
                break
            yield els

    #moves
    def __str__(self):
        return "\n".join([str(i) for i in self.get_iter()])    

    def peek(self, numElems):
        """ Inspect first `numElems` elements of the stream
            without actually reading them, so that they still
            are on top of the stream
        """                
        self._ensureNumElems(numElems)
        # Take either numElems, or the entire head if not enough elements.
        last = min(self.head.shape[0], numElems)
        return self.head[:last]

    #moves
    def read(self,numElems):
        """ Returns a numpy array of first `numElems` elements
            of the stream (they disappear from the stream).
            If the an element is already a numpy array,
            it will stack elements vertically.
        """
        self._ensureNumElems(numElems)
        # Take either numElems, or the entire head if not enough elements.
        last = min(self.head.shape[0], numElems)
        res = self.head[:last]
        self.head = self.head[last:]
        return res    

    #passes
    def map(self, func, chunk_size=1000000, *args, **kwargs):
        return Stream((func(el, *args, **kwargs)
                    for el in self.get_iter(chunk_size)))

    #passes
    def chunked(self, length, overlap):
        """ Returns a stream of 1 more dimension, consisting of chunks
            of form |overlap|length - 2*overlap|overlap|
            based on the original stream
        """
        def new_gen():
            buffer = self.read(length)
            while True:
                yield np.array([buffer]) #pack into one more dimension
                new_elems = self.read(length - overlap)
                if new_elems.shape[0] == 0:
                    # Reached the end of the stream
                    break
                buffer[:overlap] = buffer[length-overlap:]
                buffer[overlap:] = new_elems
        return Stream(new_gen())


    def merge(*args, chunk_size=100000):
        """ Returns a stream with columns corresponding to streams.
            Can be used both as `Stream.merge` and `obj.merge`,
            since it takes streams as arguments anyway.
            Ends when one of the streams ends.

            Works only with 1d or 2d data in the same format.

            When stacking arrays must be of similar shapes,
            therefore we first stack all 1d arrays, and then stack them
            with the 2d arrays.
        """
        def gen():
            while True:
                lst1dim = []
                lst2dim = []
                for stm in args:
                    els = stm.read(chunk_size)
                    if els.shape[0] == 0:
                        return
                    if els.ndim == 1:
                        lst1dim.append(els)
                    else:
                        lst2dim.append(els)
                lst2dim.append(np.column_stack(lst1dim))
                yield np.hstack(lst2dim)
        return Stream(gen())
    



def stm(n):
    return Stream(np.array([i, i+1]).T for i in range(n))


def get_simple_stream():
    return Stream(np.arange(3*i, 3*i+3) for i in range(10))

stm1 = get_simple_stream()
stm2 = Stream.merge(get_simple_stream(), get_simple_stream())
merged = Stream.merge(stm1, stm2)

# Stream can be either 1-dimensional, representing
# one channel with respect to time, or it can be
# 2-dimensional, with columns representing channels and rows representing time.
# In case of chunks, they are 1 dimension higher than the original data,
# so they can be 3-dimensional.
# The vertical direction represents time, horizontal represent channels.
# Thus n-th element of the Stream will be either a number
# or a 1-d array at time n.



