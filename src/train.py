import sys
import os
import argparse

import files_io
from generator import Generator

import pdb

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
    model = p['model']
    feats_to_use = p['feats_to_use']
    batch_size = 10

    #Lists
    tr_dir = './data/features/'+feats_to_use+'/training/'
    val_dir = './data/lists/'+feats_to_use+'/validation/'
    tst_dir = './data/lists/'+feats_to_use+'/testing/'
    scalers = './data/lists/'+feats_to_use+'/all_scalers.txt'

    #Load the features
    #Generators
    tr_datagen = Generator(tr_dir, scalers)
    val_datagen = Generator(val_dir, scalers)

    tr_datagen.flow_from_dir((15,batch_size), batch_size, 2)

    pdb.set_trace()






if __name__ == "__main__":
  main(sys.argv)
