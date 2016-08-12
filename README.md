# dspflow

A module for stream-based processing of data. 


## Installation

First make sure you have numpy and h5py installed (you should be using conda)

    conda install numpy h5py

git clone https://github.com/kylerbrown/dspflow.git
cd dspflow
python setup.py install


## Usage


    import numpy as np
    from dspflow import Stream, ArfStreamer, DatStreamer

    with ArfStreamer.open('data.arf') as arffile:
        stream = arffile.stream_channels(['entry1/channel1', 'entry2/channel2'], 
            chunk_size=1000000, verbose=True)
        stream = stream.map(lambda chunk: np.sum(chunk, axis=-1)) #last dimension is time

        DatStreamer.save(stream, 'data.dat')