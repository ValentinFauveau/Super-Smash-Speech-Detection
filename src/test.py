import os
import sys
import argparse
import numpy as np

import files_io
from generator import Generator
from statistics import mode

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
    nb_files = tst_datagen.get_nbfiles()

    #Load model
    model = files_io.load_model(model_name)

    #Testing unseen data
    predictions = []
    gt = []

    file_predictions = []
    file_class = []

    import glob
    aa = glob.glob(tst_dir+'*.feats')

    temp_gen = tst_datagen.file_flow(len(LABELS))

    for i in range(nb_files):
        X_test, y_test, file = next(temp_gen)
        preds = model.predict_classes(X_test, batch_size=batch_size, verbose=0)

        print('***>File: '+file)
        print('Frame predictions:')
        print(preds)
        print('Ground Truth:')
        print(np.argmax(y_test,axis=1))
        print('File Prediction: '+str(mode(preds)) + ' Class: '+ str(np.argmax(y_test,axis=1)[0]))
        print('\n')

        gt.append(np.argmax(y_test,axis=1))
        predictions.append(preds)
        file_class.append(np.argmax(y_test,axis=1)[0])
        file_predictions.append(mode(preds))

    gt = np.concatenate(gt)
    predictions = np.concatenate(predictions)

    tp = np.sum(np.equal(gt,predictions).astype(int))
    tp_file = np.sum(np.equal(file_class,file_predictions).astype(int))

    accuracy = tp/len(gt)
    accuracy_file = tp_file/len(file_class)

    print('Accuracy Frame Based: ' + str(np.round(accuracy,3)))
    print('Accuracy File Based: ' + str(np.round(accuracy_file,3)))

    # end of the main function


if __name__ == "__main__":
    main(sys.argv)
