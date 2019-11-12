import os
import sys
import argparse
import glob
import shutil
import numpy as np
from sklearn.utils import shuffle

import pdb

import files_io

#Constants
RND_SEED = 3



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
    tr_split = p['tr_split']
    val_split = p['val_split']
    tst_split = p['tst_split']
    lst_name = p['lst_name']

    if (tr_split+val_split+tst_split) != 1:
        print("**> ERROR: the training, validation and test split must be equal to 1")
        exit()

    #Generates the lists of sounds for each dataset
    lst_tr = []
    lst_val = []
    lst_tst = []

    labels = os.listdir(data_dir)

    for l in labels:

        files = glob.glob(os.path.join(data_dir,l)+'/*.wav')
        files = shuffle(files, random_state=RND_SEED)

        lst_tr.append(files[:int(len(files)*tr_split)])
        lst_val.append(files[int(len(files)*tr_split) : int(len(files)*(tr_split+val_split))])
        lst_tst.append(files[int(len(files)*(tr_split+val_split)):])

    #Concatenates the lists
    lst_tr = np.concatenate(lst_tr)
    lst_val = np.concatenate(lst_val)
    lst_tst = np.concatenate(lst_tst)

    #Creates the output folder for the new lists
    out = './data/lists/'+lst_name

    if os.path.exists(out):
        shutil.rmtree(out)
    try:
        os.makedirs(out)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # name of the lists
    out_tr_dir = './data/lists/'+lst_name+'/'+lst_name+'_tr.lst'
    out_val_dir = './data/lists/'+lst_name+'/'+lst_name+'_val.lst'
    out_tst_dir = './data/lists/'+lst_name+'/'+lst_name+'_tst.lst'

    #save lists
    files_io.savelist(out_tr_dir, lst_tr)
    files_io.savelist(out_val_dir, lst_val)
    files_io.savelist(out_tst_dir, lst_tst)

if __name__ == "__main__":
    main(sys.argv)
