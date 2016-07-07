
import numpy as np
import copy
import arf

def merge(*args):
    def gen():
        # TODO research why assigning here doesn't work
        # arr = np.zeros((1,len(args)))
        while True:
            arr = np.zeros((len(args)))
            try:
                for (i, arg) in enumerate(args):
                    n = next(arg)
                    arr[i] = n
                yield arr
            except StopIteration:
                break
    return Stream(gen())


class ArfStreamer():
    def open(filename, path, chunk_size):
            def gen(filename, path, chunk_size=100000):
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
                        buffer = dataset[offset:offset+chunk]
                        for b in buffer:
                            yield b
                        offset += chunk
            return Stream(gen(filename, path, chunk_size))

    # def save(stream, path, chunk_size=100000):
    #     with arf.open_file(filename, 'w+') as file:
    #         path = path.split("/")
    #         ds_name = path[-1]
    #         grp_path = "/".join(path[:-1])
    #         grp = file.create_group(grp_path)
    #         # TODO check if already exists
            
    #         #Get first batch of data
    #         data = stream.readElements(chunk_size)
    #         grp.create_dataset(ds_name, data=data)
    #         while len(data)>0:
    #             data = stream.readElements(chunk_size)

class Stream():
    def __init__(self, gen):
        self.gen = gen
        self.head = [] #Here we store elements that we peeked at

    def __next__(self):
        if len(self.head) == 0:
            return next(self.gen)
        else:
            return self.head.pop(0)

    def __iter__(self):
        return self

    #passes
    def map(self, func, *args, **kwargs):
        return Stream((func(el, *args, **kwargs)) for el in self)
    
    #passes
    def chunked(self, length, overlap):
        """ Returns a stream returning numpy arrays of form
            |overlap|length - 2*overlap|overlap|
            based on the original stream
        """
        def new_gen():
            first_chunk = True
            last_chunk = False
            buffer = []
            while not last_chunk:
                while (len(buffer) < length) and not last_chunk:
                    try:
                        n = next(self)
                        buffer.append(copy.copy(n))
                    except StopIteration:
                        last_chunk = True
                chunk = np.array(buffer)
                # Returns the last, incomplete chunk as well
                yield chunk
                buffer = buffer[-overlap:]
        return Stream(new_gen())

    def peek(self, numElems):
        """ Inspect first numElems elements of the stream
            without actually reading them, so that they still
            are on top of the stream
        """
        while len(self.head) < numElems:
            try:
                n = next(self.gen)
            except StopIteration:
                raise
            self.head.append(n)
        return self.head[:numElems]

    #moves
    def __str__(self):
        res = []
        while True:
            try:
                n = next(self)
                res.append(str(n))
            except StopIteration:
                return "\n".join(res)

    #moves
    def readElements(self,numElems):
        """ Returns an array of first numElems elements
            of the stream (they disappear from the stream)
        """
        res = []
        for i in range(numElems):
            try:
                res.append(next(self))
            except StopIteration:
                break
        return res



def stm(n):
    return Stream(i for i in range(n))


# Examples:
# > i = stm(10).map(lambda x: 2*x)
# > j = stm(10)
# > i.peek(2)
# [0,2]
# > k = merge(i,j)
# > k.peek(3)
# [array([ 0.,  0.]), array([ 2.,  1.]), array([ 4.,  2.])]
# > k.chunked(2,1).peek(2)
# [array([[ 0.,  0.],
#         [ 2.,  1.],
#         [ 4.,  2.]]), array([[ 4.,  2.],
#         [ 6.,  3.],
#         [ 8.,  4.]])]




