import os
import sys
import argparse
import numpy as np

import files_io
from generator import Generator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
    model = p['model_to_use']
    model_name = p['out_model_name']
    feats_to_use = p['feats_to_use']
    feats_dim = p['feats_dim']
    LABELS = p['LABELS']
    batch_size = p['batch_size']

    #Lists
    tst_dir = './data/features/'+feats_to_use+'/testing/'
    scalers = './data/features/'+feats_to_use+'/all_scalers.txt'

    #Load the features
    #Generators
    tst_datagen = Generator(tst_dir, scalers)

    #Dimensions of the data
    tst_nb_frames = tst_datagen.get_nbframes()
    tst_step_per_epoch = tst_nb_frames // batch_size

    #Load model
    model = files_io.load_model(model_name)

    #Testing unseen data
    predictions = []
    gt = []

    for _ in range(tst_step_per_epoch):
        X_test, y_test = next(tst_datagen.flow_from_dir(feats_dim,batch_size,len(LABELS)))
        preds = model.predict_classes(X_test, batch_size=batch_size, verbose=0)

        gt.append(np.argmax(y_test,axis=1))
        predictions.append(preds)

    gt = np.concatenate(gt)
    predictions = np.concatenate(predictions)

    tp = np.sum(gt == predictions)

    accuracy = tp/len(gt)

    print('The Accuracy of the model is: ' + str(np.round(accuracy,3)))


if __name__ == "__main__":
    main(sys.argv)
