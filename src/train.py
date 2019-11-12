import sys
import os
import argparse

from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
os.environ['CUDA_VISIBLE_DEVICES']="0"
# link: https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
    algo = p['algo']
    feats_to_use = p['feats_to_use']
    feats_dim = p['feats_dim']
    LABELS = p['LABELS']
    batch_size = p['batch_size']
    lr = p['lr']
    nepochs = p['nepochs']
    l1 = p['l1']
    out_model_name = p['out_model_name']

    #Lists
    tr_dir = './data/features/'+feats_to_use+'/training/'
    val_dir = './data/features/'+feats_to_use+'/validation/'
    tst_dir = './data/features/'+feats_to_use+'/testing/'
    scalers = './data/features/'+feats_to_use+'/all_scalers.txt'
    odir = './output/'+out_model_name+'/'

    #Load the features
    #Generators
    tr_datagen = Generator(tr_dir, scalers)
    val_datagen = Generator(val_dir, scalers)

    #Dimensions of the data
    tr_nb_frames = tr_datagen.get_nbframes()
    val_nb_frames = val_datagen.get_nbframes()
    tr_step_per_epoch = tr_nb_frames // (batch_size*2)
    val_step_per_epoch = val_nb_frames // batch_size

    #Model declaration
    model = Sequential()
    model.add(Dense(
			l1,
			input_dim=feats_dim,
			activation='relu'
		))
    model.add(Dense(
			len(LABELS),
			activation='softmax'
		))

    #Model compile
    optimizer = Adam(lr=lr)
    model.compile(loss = 'categorical_crossentropy', optimizer=optimizer,
                metrics=['accuracy'])

    print(model.summary())

    # pdb.set_trace()

    #Fit the model
    h = model.fit_generator(tr_datagen.flow_from_dir(feats_dim,batch_size,len(LABELS))
            , validation_data=val_datagen.flow_from_dir(feats_dim,batch_size,len(LABELS))
            , steps_per_epoch=tr_step_per_epoch, epochs=nepochs, validation_steps=val_step_per_epoch, verbose=1)

    model_json = model.to_json()
    ## save the model architecture
    #
    if not os.path.exists(odir):
        os.makedirs(odir)

    # pdb.set_trace()

    json_net = open(os.path.join(odir, out_model_name+'.json'), "w")
    json_net.write(model_json)
    model.save_weights(os.path.join(odir, out_model_name+'.h5'))

    ## save the model weights
    #
    model.save(os.path.join(odir, out_model_name+'.h5'), overwrite=True)






if __name__ == "__main__":
  main(sys.argv)
