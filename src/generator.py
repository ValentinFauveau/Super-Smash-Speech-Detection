import os
import glob
import random
import numpy as np
import struct

import pdb

class Generator():

    def __init__(self, data_dir, scalers_dir):

        self.data_dir = data_dir
        self.scalers_dir = scalers_dir

    def flow_from_dir(self, target_size, batch_size, num_classes):

        batch_ind = 0

        batch_feats = np.zeros((batch_size, target_size[0], target_size[1]), dtype=np.float32)
        batch_labels = np.zeros((batch_size, num_classes), dtype=np.int)

        while True:

            chunks = {}
            N_frames = 0

            for chunk in glob.glob(self.data_dir+'*.feats'):
                chunks[chunk] = list(range(self.read_nframes(chunk)))
                N_frames += self.read_nframes(chunk)

            while len(chunks) != 0:

                rnd_chunk = random.choice(list(chunks.keys()))
                rnd_frame = random.sample(chunks[rnd_chunk], 1)[0]

                if len(chunks[rnd_chunk]) == 0:
                    del chunks[rnd_chunk]

                feats, label = self.read_frame(rnd_chunk,rnd_frame)

                pdb.set_trace()



    def read_nframes(self, chunk):
        f = open(chunk, 'rb')
        nframes = struct.unpack('I', f.read(4))[0]
        return nframes


    def read_frame(self, chunk, frame_num):
        f = open(chunk, 'rb')

        pdb.set_trace()

        nframes = struct.unpack('I', f.read(4))[0]
        nfeats = struct.unpack('I', f.read(4))[0]
        f.seek(frame_num * (4 + 8 * nfeats) ,1)
        label = struct.unpack('I', f.read(4))[0]
        feats = np.fromstring(f.read(8 * nfeats), dtype=np.float32)

        return (feats,label)
