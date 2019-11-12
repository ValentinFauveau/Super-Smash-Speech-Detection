import sys
import os
import shutil
import argparse
import numpy as np

#Loads audio files
from scipy.io import wavfile

#Play sounds
from playsound import playsound

#Frequency Features
import python_speech_features as psf

import pdb

import files_io

def main(argv):

    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-parameters', type=str)
    args = parser.parse_args()

    if args.parameters is None:
        print("**> ERROR: Must provide parameter file")

    # reads all the parameters
    p = files_io.read_parameters(args.parameters)

    #Reads the parameters
    data_dir = p['data_dir']
    lst = p['lst_to_use']
    LABELS = p['LABELS']

    #Creates folders
    feats_path = "./data/features/"+lst+'/'

    if os.path.exists(feats_path):
        shutil.rmtree(feats_path)
    try:
        os.makedirs(feats_path+'training/')
        os.makedirs(feats_path+'validation/')
        os.makedirs(feats_path+'testing/')

    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    #Lists
    lst_tr = './data/lists/'+lst+'/'+lst+'_tr.lst'
    lst_val = './data/lists/'+lst+'/'+lst+'_val.lst'
    lst_tst = './data/lists/'+lst+'/'+lst+'_tst.lst'

    lists = [lst_tr, lst_val, lst_tst]

    #Loops over the lists, calculates and saves the features
    for l in lists:

        #Check if the list correspond to the training list to calculate the mean
        #and std for every feature for standarization.
        if l == lst_tr:
            means = 0
            stds = 0
            N = 0

        lst_files = files_io.read_list(l)

        for sample in lst_files:

            #loads the wav file
            fs, data = wavfile.read(sample)

            if data.ndim > 1:
                data = data[:,1]

            label = LABELS[sample.split(os.path.sep)[-2]]

            try:
                mfccs = extract_mfccs(data,fs,p)
            except:
                pdb.set_trace()
            energy = extract_energy(data,fs,p)
            length = extract_length(data,fs,p)

            feats = np.column_stack((mfccs,energy,length))
            labels = [label]*feats.shape[0]

            #Saves means and stds for every feature on the training list
            if l == lst_tr:
                means += np.sum(feats, axis=0).astype(np.float64)
                stds += np.sum(np.square(feats), axis=0).astype(np.float64)
                N += feats.shape[0]

            #Saves feats and labels into binary files
            if l == lst_tr:
                dirname = "./data/features/"+lst+'/'+'training/'+os.path.basename(sample)[:-4]+'.feats'
            elif l == lst_val:
                dirname = "./data/features/"+lst+'/'+'validation/'+os.path.basename(sample)[:-4]+'.feats'
            else:
                dirname = "./data/features/"+lst+'/'+'testing/'+os.path.basename(sample)[:-4]+'.feats'

            files_io.save_feats_labs(dirname, feats, labels)


        if l == lst_tr:
            means = (means/N).astype(np.float32)
            stds = (np.sqrt(stds / N - np.square(means))).astype(np.float32)

            odir = "./data/features/"+lst+'/'+lst+'_scalers.txt'
            files_io.save_means_stds(odir ,means, stds)




#Extracts MFCCs from sound segments
def extract_mfccs(data, fs, p):

    mfccs = psf.mfcc(data, samplerate=fs, winlen=p['winlen'],
			winstep=p['winstep'], numcep=int(p['nfilt']/2),
			nfilt=p['nfilt'], nfft=p['nfft'], lowfreq=0,
			highfreq=fs/2).astype(np.float32)

    return mfccs


#Extracts energy from sound segments
def extract_energy(data, fs, p):

    samplesperwin = int(round(fs*p['winlen']))
    nwins = int(np.ceil(data.shape[0] / samplesperwin))
    out = np.zeros((nwins, 1), dtype=np.float64)

    # loops through the windows
    for i in range(nwins):

        st = i * samplesperwin
        en = (i + 1) * samplesperwin
        power = np.sum(np.square(data[st:en]))
        energy = power / samplesperwin
        out[i] = np.array(energy)

        if np.isnan(out[i]):
            pdb.set_trace()

    # take the log so value will fit in float32
    # return np.log(out).astype(np.float32)
    return out


#Extracts the length of the samples
def extract_length(data, fs, p):

    samplesperwin = int(round(fs*p['winlen']))
    nwins = int(np.ceil(data.shape[0] / samplesperwin))
    sample_length = len(data)/fs
    out = np.ones((nwins, 1))*sample_length

    return out


if __name__ == "__main__":
    main(sys.argv)
