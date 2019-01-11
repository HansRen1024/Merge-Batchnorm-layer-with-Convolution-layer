import numpy as np  
import sys
caffe_root = '/home/hans/caffe-FaceBoxes/'
sys.path.insert(0, caffe_root + 'python')  
import caffe  

train_proto = 'train.prototxt'  # with bn layers
train_model = 'faceboxes_iter_gray.caffemodel'  #should be your snapshot caffemodel

deploy_proto = 'faceboxes_gray_deploy.prototxt'   # without bn layers
save_model = 'faceboxes_gray_deploy.caffemodel' # output model

def merge_bn_mobilenetssd(net, nob):
    '''
    merge the batchnorm, scale layer weights to the conv layer, to  improve the performance
    var = var + scaleFacotr
    rstd = 1. / sqrt(var + eps)
    w = w * rstd * scale
    b = (b - mean) * rstd * scale + shift
    '''
    for key in net.params.iterkeys():
        if type(net.params[key]) is caffe._caffe.BlobVec:
            if key.endswith("_bn") or key.endswith("_scale"):
                continue
            else:
                conv = net.params[key]
                if not net.params.has_key(key + "_bn"):
                    for i, w in enumerate(conv):
                        nob.params[key][i].data[...] = w.data
                else:
                    bn = net.params[key + "_bn"]
                    scale = net.params[key + "_scale"]
                    wt = conv[0].data
                    channels = wt.shape[0]
                    bias = np.zeros(wt.shape[0])
                    if len(conv) > 1:
                        bias = conv[1].data
                    mean = bn[0].data
                    var = bn[1].data
                    scalef = bn[2].data

                    scales = scale[0].data
                    shift = scale[1].data

                    if scalef != 0:
                        scalef = 1. / scalef
                    mean = mean * scalef
                    var = var * scalef
                    rstd = 1. / np.sqrt(var + 1e-5)
                    rstd1 = rstd.reshape((channels,1,1,1))
                    scales1 = scales.reshape((channels,1,1,1))
                    wt = wt * rstd1 * scales1
                    bias = (bias - mean) * rstd * scales + shift
                    
                    nob.params[key][0].data[...] = wt
                    nob.params[key][1].data[...] = bias
def merge_bn_faceboxes(net, nob):
    layer_names = []
    num=1
    for layer in net.params.iterkeys():
        if type(net.params[layer]) is caffe._caffe.BlobVec:
            layer_names.append(layer)
    for ind, key in enumerate(layer_names):
        if type(net.params[key]) is caffe._caffe.BlobVec:
            if ind<6:
                for i, w in enumerate(net.params[key]):
                    nob.params[key][i].data[...] = w.data
                continue
            if key.endswith("_bn") or key.endswith("_scale"):
                continue
            else:
                conv = net.params[key]
                if key.endswith("_loc") or key.endswith("_conf"):
                    for i, w in enumerate(conv):
                        nob.params[key][i].data[...] = w.data
                else:
                    print("%d processing conv layer: %s" %(num, key))
                    num+=1
                    bn = net.params[key + "_bn"]
                    scale = net.params[key + "_scale"]
                    wt = conv[0].data
                    channels = wt.shape[0]
                    bias = np.zeros(wt.shape[0])
                    if len(conv) > 1:
                        bias = conv[1].data
                    mean = bn[0].data
                    var = bn[1].data
                    scalef = bn[2].data

                    scales = scale[0].data
                    shift = scale[1].data

                    if scalef != 0:
                        scalef = 1. / scalef
                    mean = mean * scalef
                    var = var * scalef
                    rstd = 1. / np.sqrt(var + 1e-5)
                    rstd1 = rstd.reshape((channels,1,1,1))
                    scales1 = scales.reshape((channels,1,1,1))
                    wt = wt * rstd1 * scales1
                    bias = (bias - mean) * rstd * scales + shift
                    
                    nob.params[key][0].data[...] = wt
                    nob.params[key][1].data[...] = bias
net = caffe.Net(train_proto, train_model, caffe.TRAIN)  
net_deploy = caffe.Net(deploy_proto, caffe.TEST)  

#merge_bn_mobilenetssd(net, net_deploy)
merge_bn_faceboxes(net, net_deploy)
net_deploy.save(save_model)

