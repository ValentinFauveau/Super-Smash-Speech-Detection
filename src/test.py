import os
import sys
import argparse

import pdb

def main():

    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-parameters', type=str)
    args = parser.parse_args()

    if args.parameters is None:
        print("**> ERROR: Must provide parameter file")

    # reads all the parameters
    p = files_io.read_parameters(args.parameters)

    #Reads the parameters
    model = p['model_to_use']
    model_name = p['out_model_name']
    feats_to_use = p['feats_to_use']
    feats_dim = p['feats_dim']
    LABELS = p['LABELS']

    #Lists
    tst_dir = './data/features/'+feats_to_use+'/testing/'
    scalers = './data/features/'+feats_to_use+'/all_scalers.txt'

    #Load the features
    #Generators
    tst_datagen = Generator(tst_dir, scalers)

    #Load model
    model = files_io.load_model(model_name)


if __name__ == "__main__":
    main(sys.argv)
