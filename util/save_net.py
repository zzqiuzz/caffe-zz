
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
import re
import code
import argparse
import scipy.io as io
import caffe
from google.protobuf import text_format


def check_file(filename):
    assert os.path.isfile(filename), "%s is not a file" % filename


def load_net(model,weights):
    ''' load the network definition from the models directory
        Input:
            net_name -- name of network as used in the models directory
        Returns:
            net      -- caffe network object
    '''
    net = caffe.Net(model, weights, caffe.TEST)

    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    mean_file = caffe_root + '/python/caffe/imagenet/ilsvrc_2012_mean.npy'
    check_file(mean_file)
    transformer.set_mean('data', np.load(mean_file).mean(1).mean(1)) # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

    # set net to batch size of 50
    batch_size = 50
    net.blobs['data'].reshape(batch_size,3,224,224)

    return net


def read_prototxt(model):
    from caffe.proto import caffe_pb2
    net_param = caffe_pb2.NetParameter()

    print 'reading prototxt',model
    with open(model) as f:
        text_format.Merge(str(f.read()), net_param)

    return net_param


def roundup_to_multiple(x, m):
    return int(np.ceil( x / float(m))) * m

def divide_roundup(x, m):
    return roundup_to_multiple(x,m) / m

def to2d(a):
    a = np.array(a)
    return np.reshape(a, (a.shape[0],-1))


def write_trace(net, layers, batches, dir,save_out_layer_name):
    ''' runs the network for a specified number of batches and saves the inputs to each layer
        Inputs:
            net -- caffe net object
            layers -- vector of protobuf layers to save
            batches  -- number of batches to run
            dir -- directory to write trace files
        Returns:
            nothing           
    '''

    for b in range(batches):
        print "iteration %d" % b

        start = time.time()
        net.forward()
        end = time.time()
        print 'runtime: %.2f' % (end-start)
 

        print 'layer, Nb, Ni, Nx, Ny'
        for l, layer in enumerate(layers):
            name = layer.name
            if name in save_out_layer_name:
                sane_name = re.sub('/','-',name) # sanitize layer name so we can save it as a file (remove /)
                savefile = '%s/%s-%d' % (dir, sane_name, b)

                '''if os.path.isfile(savefile + ".npy"):
                    print savefile, "exists, skipping"
                    continue'''

                if not os.path.exists(dir):
                    os.makedirs(dir)

                input_blob = layer.top[0]
                data = net.blobs[input_blob].data
                print data.shape
                data = data.reshape([1,data.size])
                if (len(data.shape) == 2):
                    (Nb, Ni) = data.shape
                    data = data.reshape( (Nb,Ni,1,1) )
            
                print "saving: ", savefile
                #np.save(savefile,data)
                io.savemat(savefile, {'name': data})
                


def write_config(net, layers, model, weights, dir):
    ''' write a set of layer parameters for each layer
        Input:
            net -- caffe net object
            layers -- vector of layers protobufs to save
            model -- model prototxt filepath
            weights -- model weight filepath
            dir -- output directory
        Returns:
            nothing
    '''

    file_handle = open(dir + "/trace_params.csv", 'w')
    print "layer, input, Nn, Kx, Ky, stride, pad"
    for l, layer in enumerate(layers):
        name = layer.name
        sane_name = re.sub('/','-',name) # sanitize layer name so we can save it as a file (remove /)

        input_blob = layer.bottom[0] # assume conv always has one input blob
        stride = layer.convolution_param.stride
        pad = layer.convolution_param.pad

        data = net.blobs[input_blob].data
        weights = net.params[name][0].data
        print name, "D", data.shape, "W", weights.shape

        if (len(weights.shape) == 2):
            (Nn, Ni) = weights.shape
            (Kx, Ky) = (1,1)
        else:
            (Nn, Ni, Kx, Ky) = weights.shape

        if (len(data.shape) == 2):
            (Nb, Ni) = data.shape
            (Nx, Ny) = (1,1)
        else:
            (Nb, Ni, Nx, Ny) = data.shape
            #(Kx, Ky) = (Nx, Ny)

        outstr = ','.join( [str(i) for i in [name, input_blob, Nn, Kx, Ky, stride, pad]] ) + "\n"
        print outstr
        file_handle.write(outstr)
    file_handle.close()

caffe_root = os.getenv('CAFFE_ROOT') 
'''
parser = argparse.ArgumentParser(prog='save_net.py', description='Run a network in pycaffe and save a trace of the data input to each layer')
parser.add_argument('--batches', metavar='batches', type=int, help='batches to run')
parser.add_argument('--skip', type=int,   default=0,          help='batches to skip')
parser.add_argument('-o'    , type=str,   default=trace_dir,  help='output directory for trace files')
parser.add_argument('-p'    , dest='write_params', action='store_true', help='write layer parameters for each net instead of writing trace')
parser.set_defaults(write_params=False)

args = parser.parse_args()
batches = args.batches 
skip    = args.skip
write_params = args.write_params
trace_dir = args.o'''

batches = 1
write_params = False
out_dir = './'
caffe.set_device(6)
caffe.set_mode_gpu()
save_out_layer_name = ['conv1_dw']#['bn3','scale3','relu3','conv4']
model = caffe_root + "/util/MobileFaceNet_deploy_mergebn_relu_pooling.prototxt"
weights = caffe_root + "/util/MobileFaceNet_96_96_iter_400000_mergebn_relu.caffemodel"
check_file(model)
check_file(weights)

net = load_net(model,weights)
net_param = read_prototxt(model)
if write_params:
    layers = [l for l in net_param.layer if l.type in ['Convolution','InnerProduct']]
    write_config(net, layers, model, weights, out_dir)
else:
    layers = [ l for l in net_param.layer ]
    write_trace(net, layers, batches, out_dir,save_out_layer_name)
    
