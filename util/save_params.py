import caffe
import numpy as np
import matplotlib.pyplot as plt
import os
from caffe.proto import caffe_pb2
net_param = caffe_pb2.NetParameter()
with open(model) as f:
        text_format.Merge(str(f.read()), net_param)
caffe_root = os.getenv('CAFFE_ROOT')
model_file = caffe_root + "/my/Imagenet/AlexNet-BN/FWN/train_val_deploy.prototxt"
weights_file = caffe_root + "/my/Imagenet/AlexNet-BN/FWN/bvlc_alexnet_bn_shrt.caffemodel"
save_weight_layer_name = 'conv3'
save_out_layer_name = 'bn3'
caffe.set_device(1)  # if we have multiple GPUs, pick the first one
caffe.set_mode_gpu()
net = caffe.Net(model_file,weights_file,caffe.TEST)

mu = np.load(caffe_root + "/python/caffe/imagenet/ilsvrc_2012_mean.npy")
mu = mu.mean(1).mean(1) 

transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
batch_size = 1
net.blobs['data'].reshape(batch_size,3,224,224) 

for b in range(batch_size):
    print 'iteration: ' + str(b)
    net.forward()
    for layer_name in list(net._layer_names):
        if save_out_layer_name == layer_name:


###save specified layer params
for param_name in net.params.keys():
    if save_weight_layer_name == param_name:
        weight = net.params[param_name][0].data
        np.save(param_name,weight)
        break


