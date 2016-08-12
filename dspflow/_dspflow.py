import numpy as np
import copy
import arf

class ArfStreamer():
    """ Wraps a h5py.File object. Should be used only inside a with statement
        so that it's never left opened after the program. 
        You can access the File methods using `arfstreamer.file`
    """
    def __init__(self, filename):
        self.filename = filename

    def __enter__(self):
        self.file = arf.open_file(self.filename)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.file.close()

    
    def stream_channel(self, path, chunk_size=1000000, num_samples=None, verbose=False):
        """ Returns a Stream object wrapping the dataset at `path`.
            `chunk_size` is the size of the chunk in which values are yielded;
            the bigger the better, as long as it fits in the memory.
            If `num_samples` is specified, it returns up to that many elements.
        """
        def gen(path, chunk_size):
            dataset = self.file[path]
            if num_samples != None:
                length = num_samples
            else:
                length = len(dataset)
            offset = 0
            while offset < length:
                if length - offset > chunk_size:
                    chunk = chunk_size
                else:
                    chunk = length - offset
                verbose and print("processing new batch {0} of {1} ({2:.2%})".format(
                    offset, length, offset/length))
                buffer = dataset[offset:offset+chunk]
                yield buffer
                offset += chunk
        return Stream(gen(path, chunk_size), chunk_size=chunk_size)


    def stream_channels(self, paths, chunk_size=10000000, verbose=False):
        """ Return a Stream object wrapping a concatenation od datasets at `paths`.
            Each dataset must be 1-d or 2-d (if 2-d, then time is the vertical axis),
            and they are stacked as columns. In theory should be faster than
            multiple `stream_channel` and `merge` calls, but depends on the number of datasets
            and on the `chunk_size`.
        """
        def gen(paths, chunk_size):
            datasets = [self.file[path] for path in paths]
            length = len(datasets[0]) # Assume all are the same size
            offset = 0
            col_sizes = [dataset.shape[0] if dataset.ndim>1 else 1 for dataset in datasets]
            total_cols = sum(col_sizes)
            buf = None
            while offset < length:
                if length - offset > chunk_size:
                    chunk = chunk_size
                    if buf == None:
                        buf = np.zeros((chunk, total_cols))
                else:
                    chunk = length - offset
                    buf = np.zeros((chunk, total_cols))
                verbose and print("batch {0} of ({2:.2%}) from {3} datasets".format(
                    offset, length, offset/length, len(datasets)))
                running_col = 0
                for (i, dst) in enumerate(datasets):
                    num_cols = col_sizes[i]
                    read_values = dst[offset:(offset+chunk)]
                    read_values.resize((chunk, num_cols))
                    buf[:, running_col:(running_col + num_cols)] = read_values
                    running_col += num_cols
                yield buf
                offset += chunk
        return Stream(gen(paths, chunk_size), chunk_size=chunk_size)
                    


    def save(stream, filename, path, sampling_rate=None, chunk_size=None):
        """ Saves a Stream object to an .arf file.
            Can't be called by an instance of ArfStreamer.
        """
        if chunk_size == None:
            chunk_size = stream.chunk_size

        if sampling_rate == None:
            raise Exception("You must specify the sampling rate in ArfStreamer.save")
        
        with arf.open_file(filename, 'a') as file:
            path = path.split("/")
            dst_name = path[-1]
            grp_path = "/".join(path[:-1])
            grp = file.require_group(grp_path)            
            #Get first batch of data
            data = stream.read(chunk_size)
            try:
                dst = arf.create_dataset(grp, dst_name, data,
                    maxshape=(None,), sampling_rate=sampling_rate)
            except:
                raise ValueError('Error, maybe dataset with that name already exists')
            while True:
                data = stream.read(chunk_size)
                if len(data) == 0:
                    break
                arf.append_data(dst, data)
            file.flush()


class DatStreamer():
    def save(stream, filename, truncate=True):
        if truncate:
            with open(filename, 'w+b') as f:
                pass

        for chunk in stream.get_iter():
            # Repeatedly open the file in the appending mode,
            # because numpy doesn't have an easy append function
            with open(filename, 'a+b') as f:
                chunk.tofile(f)

class Stream():
    def __init__(self, gen, chunk_size=1000000, head=None):
        self.gen = gen
        # Head stores the chunk we get from the generator,
        # so that we can peek or read however many elements we want.
        if head != None:
            self.head = head
        else:
            # We take a chunk immediately, to learn the shape.
            self.head = next(self.gen)

        self.chunk_size = chunk_size

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

    def get_iter(self, numPerIter=None):
        """ Return an iterator for the stream, `numPerIter` is
            the size of the first dimension of the array returned.
        """
        if numPerIter == None:
            numPerIter = self.chunk_size
        while True:
            els = self.read(numPerIter)[:]
            if els.shape[0] == 0:
                break
            yield els

    #moves
    def __str__(self):
        return "\n".join([str(i) for i in self.get_iter()])    

    def peek(self, numElems):
        """ Returns a numpy array of first `numElems` elements of the stream
            without consuming them from the stream.
        """                
        self._ensureNumElems(numElems)
        # Take either numElems, or the entire head if not enough elements.
        last = min(self.head.shape[0], numElems)
        return self.head[:last]

    #moves
    def read(self,numElems):
        """ Consumes first `numElems` elements of the stream and returns
            a numpy array with them. 1d data is concatenated, 2d data is
            stacked vertically.
        """
        self._ensureNumElems(numElems)
        # Take either numElems, or the entire head if not enough elements.
        last = min(self.head.shape[0], numElems)
        res = self.head[:last]
        self.head = self.head[last:]
        return res    

    #passes
    def map(self, func, chunk_size=None, *args, **kwargs):
        if chunk_size == None:
            chunk_size = self.chunk_size
        return Stream((func(el, *args, **kwargs)
                    for el in self.get_iter(chunk_size)), chunk_size=chunk_size)

    #passes
    def chunked(self, length, overlap):
        """ Returns a stream of 1 more dimension, consisting of chunks
            of form |overlap|length - 2*overlap|overlap|
            based on the original stream.
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
        return Stream(new_gen(), chunk_size=1)


    def merge(*args, chunk_size=None):
        """ Returns a stream with columns corresponding to streams passed to it.
            Can be used both as `Stream.merge` and `obj.merge`,
            since it takes streams as arguments anyway.
            
            Ends when one of the streams ends. 
            Works only with 1-d and 2-d data where vertical direction is time.
        """
        if chunk_size == None:
            chunk_size = args[0].chunk_size
        def gen():
            while True:
                # When stacking, arrays must be of similar shapes,
                # therefore we first stack all 1d arrays, and then stack them
                # with the 2d arrays.
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
        return Stream(gen(), chunk_size=chunk_size)
    


# Stream can be either 1-dimensional, representing
# one channel with respect to time, or it can be
# 2-dimensional, with columns representing channels and rows representing time.
# In case of chunks, they are 1 dimension higher than the original data,
# so they can be 3-dimensional.
# The vertical direction represents time, horizontal represent channels.
# Thus n-th element of the Stream will be either a number
# or a 1-d array at time n.



