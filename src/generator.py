import os
import glob
import random
import numpy as np
import struct

import files_io

import pdb

class Generator():

    def __init__(self, data_dir, scalers_dir=None):

        self.data_dir = data_dir
        self.scalers_dir = scalers_dir

        if self.scalers_dir != None:
            self.scalers = files_io.load_scalers(self.scalers_dir)

    def flow_from_dir(self, target_size, batch_size, num_classes):

        batch_ind = 0

        batch_feats = np.zeros((batch_size, target_size), dtype=np.float32)
        batch_labels = np.zeros((batch_size, num_classes), dtype=np.int)

        while True:

            chunks = {}
            N_frames = 0
            index = 0

            for chunk in glob.glob(self.data_dir+'*.feats'):
                chunks[chunk] = list(range(self.read_nframes(chunk)))
                N_frames += self.read_nframes(chunk)

            while len(chunks) != 0:

                rnd_chunk = random.choice(list(chunks.keys()))
                rnd_frame = random.sample(chunks[rnd_chunk], 1)[0]

                chunks[rnd_chunk].remove(rnd_frame)

                if len(chunks[rnd_chunk]) == 0:
                    del chunks[rnd_chunk]

                feats, label = self.read_frame(rnd_chunk,rnd_frame)

                if self.scalers_dir != None:
                    feats = self.scale_values(feats, self.scalers_dir)

                if index == batch_size-1:
                    batch_feats[index] = feats
                    batch_labels[index] = self.one_hot_encoder(label, num_classes)
                    index = 0
                    yield (batch_feats , batch_labels)
                else:
                    batch_feats[index] = feats
                    batch_labels[index] = self.one_hot_encoder(label, num_classes)
                    index += 1


    def get_nbframes(self):
        N_frames = 0
        for chunk in glob.glob(self.data_dir+'*.feats'):
            N_frames += self.read_nframes(chunk)
        return N_frames

    def read_nframes(self, chunk):
        f = open(chunk, 'rb')
        nframes = struct.unpack('I', f.read(4))[0]
        return nframes

    def one_hot_encoder(self, label, num_classes):
        l = np.zeros(num_classes, dtype=np.int)
        l[label-1] = 1
        return l

    def read_frame(self, chunk, frame_num):
        Lab_size = 4 #bytes
        Feat_size = 4 #bytes

        f = open(chunk, 'rb')

        nframes = struct.unpack('I', f.read(4))[0]
        nfeats = struct.unpack('I', f.read(4))[0]
        f.seek(frame_num * (Lab_size + Feat_size * nfeats) ,1)
        label = struct.unpack('I', f.read(Lab_size))[0]
        feats = np.fromstring(f.read(Feat_size * nfeats), dtype=np.float32)

        return (feats,label)

    def scale_values(self, feats, scalers_dir):
        scalers = self.scalers
        feats_scl = np.zeros(np.shape(feats))

        for i in range(len(feats)):
            #Z score
            feats_scl[i] = (feats[i]-scalers[i][0])/scalers[i][1]

        return feats_scl
