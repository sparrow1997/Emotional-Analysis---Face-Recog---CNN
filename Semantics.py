import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

dataset = pd.read_csv('../input/fer2013.csv')
target = dataset['emotion']
column_name = ['pixel' + (str)(i) for i in range(2304)]

one_hot =[[0 for i in range(7)] for i in range(len(target))] 
for i in range(len(target)):
    one_hot[i][target[i]]=1
target = one_hot

df = pd.DataFrame(dataset.pixels.str.split(' ',2304).tolist(),
                                   columns = column_name)

df = df.astype(float)

df = df.as_matrix()

k =df[1881]*23
temp = np.reshape(k,[48,48])
plt.imshow(temp)

def weights(shape):
    return tf.Variable(tf.truncated_normal(shape , stddev = 0.5))

def bias(len):
    return tf.Variable(tf.constant(0.05,shape = [len]))

image_size = 48
image_flat = 48*48
image_tuple = (48,48)
num_classes = 7
input_channels = 1
filter_size = 4#48*5/28
num_filters_1 = 16
num_filters_2 = 32
num_perceptrons = 999

def convolution(input,input_channels,filter_size,num_filters,do_pooling = True):
    shape = [filter_size,filter_size,input_channels,num_filters]
    w = weights(shape = shape)
    b = bias(num_filters)
    
    #Calling the predefined Convolution Fn in TensorFlow
    layer = tf.nn.conv2d(input = input,
                         filter = w,
                         strides = [1,1,1,1],
                         padding = 'SAME'
                         )
    layer += b
    layer = tf.nn.relu(layer)
    if do_pooling:
        pooled_layer = tf.nn.max_pool(value=layer,
                                      ksize = [1,3,3,1],
                                      strides = [1,3,3,1],
                                      padding = 'SAME'
                                        )
    return layer,w

def flatten(layer):
    layer_shape = layer.get_shape()
    num_feats = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer,[-1,num_feats])
    return layer_flat,num_feats

def fully_connected(input,num_inputs,num_outputs,do_sigmoid = True):
    w = weights([num_inputs,num_outputs])
    b = bias(num_outputs)
    layer = tf.matmul(input,w)+b
    
    if do_sigmoid:
        layer = tf.nn.sigmoid(layer)
    return layer

#PlaceHolders:

x = tf.placeholder(tf.float32,[None,image_flat])
x_img = tf.reshape(x,[-1,image_size,image_size,input_channels])

y_true = tf.placeholder(tf.float32,[None,num_classes])
y_true_cls = tf.argmax(y_true,dimension =1)

layer1,weight1 = convolution(input=x_img,
                    input_channels= input_channels,
                    filter_size = filter_size,
                    num_filters = num_filters_1,
                    do_pooling = True)
layer2,weight2 = convolution(input=layer1,
                    input_channels= num_filters_1,
                    filter_size = filter_size,
                    num_filters = num_filters_2,
                    do_pooling = True)

print("Layer1 - > " ,layer1)
print("Layer2 - > " ,layer1)
print("Weights1 - > " ,weight1)
print("Weights2 - > " ,weight2)
flattened,num_features = flatten(layer2)

print("Flattened Layer2 - > " ,flattened)
print("Number of Inputs to Fully Connected - > " ,num_features)

fully_connected_1 = fully_connected(input = flattened,
                                    num_inputs = num_features,
                                    num_outputs = num_perceptrons,
                                    do_sigmoid = True)
print("Fully_Connected 1 - > " ,fully_connected_1)

fully_connected_2 = fully_connected(input = fully_connected_1,
                                    num_inputs = num_perceptrons,
                                    num_outputs = num_classes,
                                    do_sigmoid = True)

print("Fully_Connected 2 - > " ,fully_connected_2)

y_pred = tf.nn.softmax(fully_connected_2)
y_pred_cls = tf.argmax(y_pred,dimension = 1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = fully_connected_2,labels=y_true)

cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate = 1e-8).minimize(cost)

correct_prediction =  tf.equal(y_true_cls,y_pred_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session = tf.Session()
session.run(tf.global_variables_initializer())

xtrain = df[:20000]
ytrain = target[:20000]
xtest = df[20000:25000]
ytest = target[20000:25000]

batch_size = 50

total_iterations = 0
def run_the_code(num_iterations):
    global total_iterations

    ctr = 0
    for i in range(total_iterations,total_iterations+num_iterations):
        x_train = xtrain[ctr:ctr+batch_size]
        y_train = ytrain[ctr:ctr+batch_size]
        ctr += batch_size
        
        feed_dict_train = {x:x_train,
                           y_true:y_train
                          }
        session.run(optimizer,feed_dict = feed_dict_train)
        acc = session.run(accuracy, feed_dict=feed_dict_train)
        print( i+1  ,acc)

for i in range(10):
    run_the_code(400)
xtrain[0].shape




    
    