import os
import struct
from ast import literal_eval
from keras.models import model_from_json

import pdb

# Reads the parameter file into a dictionary
def read_parameters(parameters):

    d = {}

    # make sure we can open the file
    parameters = check_path(parameters)

    # loop through parameter file
    for line in open(parameters, 'r'):

        # remove newlines and excess whitespace
        l = line.rstrip('\n').rstrip('\r').strip()

        if len(l) == 0 or l.startswith('#'):
            continue

        # parse the key value pair
        try:
            [key, value] = l.split('=')
        except ValueError:
            print("**> ERROR: Broken Key Value Pair (%s) in %s" \
                % (line, parameters))

        # store the key value pair
        d[key.strip()] = literal_eval(value.strip())
    return d

#Reads lists
def read_list(lst):
    out_lst = [line.rstrip('\r').rstrip('\n') for line in open(lst, 'r')]
    return out_lst

#Checks the path
def check_path(p):

    p = os.path.expanduser(p)

    # make sure file exists on disk
    if not os.path.exists(p):
        print("**>ERROR: Path Does Not Exist (%s)" % (p))
        exit(-1)

    return p

#Saves lists into
def savelist(odir, lst):
    f = open(odir, 'w')
    for item in lst:
        f.write("%s\n" % item)
    f.close()

#Saves means and stds
def save_means_stds(odir, means, stds):
    means_stds = open(odir, 'w')
    means_stds.write("Mean,StandardDeviation")
    means_stds.write("\n")
    for i in range(means.shape[0]):
        means_stds.write("%.3f,%.3f" % (means[i], stds[i]))
        means_stds.write("\n")
    means_stds.close()

#Saves feats and labels
def save_feats_labs(odir, feats, labs):

    f = open(odir, 'wb')

    f.write(struct.pack('I', feats.shape[0]))
    f.write(struct.pack('I', feats.shape[1]))

    for i in range(feats.shape[0]):
        f.write(struct.pack('I', labs[i]))
        f.write(feats[i].tostring())

    f.close()

def load_scalers(scalers_dir):
    f = open(scalers_dir, 'r')
    f.readline()

    out = []

    for line in f.readlines():

        line = line.strip()
        line = line.split(',')

        mean = float(line[0])
        std = float(line[1])

        out.append([mean,std])

    return out

def load_model(model_name):
    # load json and create model
    json_file = open('./output/'+model_name+'/'+model_name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights('./output/'+model_name+'/'+model_name+'.h5')

    return model
