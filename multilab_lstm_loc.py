import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from scipy import stats
from tensorflow.python.ops import rnn, rnn_cell
#from sklearn.metrics import roc_auc_score

def read_data(file_path):
    data = pd.read_csv(file_path,header = 0)
    return data

def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += (window_size / 2)
 
def extract_segments_alt(data, window_size = 520):
    segments = np.empty((0,(window_size + 1)))
    labels = np.empty((0))
    for i in range(0, window_size):
        signal = np.asarray(data.iloc[i,0:520])
        segments = np.vstack([segments, signal])
    labels = np.asarray(data["BUILDINGID"].map(str) + data["FLOOR"].map(str))
    labels = np.asarray(pd.get_dummies(labels))
    return segments, labels
	
def extract_segments(data, window_size = 30):
    segments = np.asarray(data.iloc[:,0:520])
    segments[segments == 100] = -110
    segments = (segments - segments.mean()) / segments.var()

    labels = np.asarray(data["BUILDINGID"].map(str) + data["FLOOR"].map(str))
    labels = np.asarray(pd.get_dummies(labels))
	#segments = np.empty((0,(window_size + 1)))
    #labels = np.empty((0))
    #for (start,end) in windows(data,window_size):
    #    if(len(data.ix[start:end]) == (window_size + 1)):
    #        signal = data.ix[start:end]["<FEATURE COLUMN NAME>"]
    #        segments = np.vstack([segments, signal])
    #        labels = np.append(labels,stats.mode(data.ix[start:end]["<CLASS COLUMN NAME>"])[0][0])
    return segments, labels

#####################################################

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.0, shape = shape)
    return tf.Variable(initial)

def LSTM(x, weight, bias):
    cell = rnn_cell.LSTMCell(n_hidden,state_is_tuple = True)
    multi_layer_cell = tf.nn.rnn_cell.MultiRNNCell([cell] * 2)
    output, state = tf.nn.dynamic_rnn(multi_layer_cell, x, dtype = tf.float32)
    output_flattened = tf.reshape(output, [-1, n_hidden])
    output_logits = tf.add(tf.matmul(output_flattened,weight),bias)
    output_all = tf.nn.sigmoid(output_logits)
    output_reshaped = tf.reshape(output_all,[-1,n_steps,n_classes])
    output_last = tf.gather(tf.transpose(output_reshaped,[1,0,2]), n_steps - 1)  
    #output = tf.transpose(output, [1, 0, 2])
    #last = tf.gather(output, int(output.get_shape()[0]) - 1)
    #output_last = tf.nn.sigmoid(tf.matmul(last, weight) + bias)
    return output_last, output_all
##################################################
win_size = 520
'''
MIMIC-III dataset can possibly be use to train and test the model. 
But beware this is not the data set used by the authors of the paper. 
For dataset description and format please see Section 3: Data Description in the paper.
'''
data = read_data("trainingData.csv")
segments,labels = extract_segments_alt(data, win_size)
#labels = np.asarray(pd.get_dummies(labels), dtype = np.int8)
# reshape to [?, n_input, 1] to [?, n_input]
# x = tf.reshape(x, [-1, n_input])
#reshaped_segments = segments.reshape([-1,len(segments),(win_size)])
reshaped_segments = segments.reshape([len(segments),(win_size + 1),1])

train_test_split = np.random.rand(len(reshaped_segments)) < 0.80
train_x = reshaped_segments[train_test_split]
train_y = labels[train_test_split]
test_x = reshaped_segments[~train_test_split]
test_y = labels[~train_test_split]

tf.reset_default_graph()

learning_rate = 0.001
training_epochs = 100
batch_size = 10
total_batches = (train_x.shape[0]//batch_size)

n_input = 520
n_steps = 10
#n_hidden = 64
n_hidden = 520
n_classes = labels.shape[1]

alpha = 0.5

x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])
y_steps = tf.placeholder(tf.float32, [None, n_classes])



#################################################
weight = weight_variable([n_hidden,n_classes])
bias = bias_variable([n_classes])
y_last, y_all = LSTM(x,weight,bias)

#all_steps_cost=tf.reduce_mean(-tf.reduce_mean((y_steps * tf.log(y_all))+(1 - y_steps) * tf.log(1 - y_all),reduction_indices=1))
all_steps_cost = -tf.reduce_mean((y_steps * tf.log(y_all))  + (1 - y_steps) * tf.log(1 - y_all))
last_step_cost = -tf.reduce_mean((y * tf.log(y_last)) + ((1 - y) * tf.log(1 - y_last)))
loss_function = (alpha * all_steps_cost) + ((1 - alpha) * last_step_cost)

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss_function)

# Model evaluation
correct_prediction = tf.equal(tf.argmax(y_last,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as session:
    tf.global_variables_initializer().run()
    for epoch in range(training_epochs):
        for b in range(total_batches):    
            offset = (b * batch_size) % (train_y.shape[0] - batch_size)
            batch_x = train_x[offset:(offset + batch_size), :]
            batch_y = train_y[offset:(offset + batch_size), :]
            batch_y_steps = np.tile(batch_y,((train_x.shape[1]),1))
            _, c = session.run([optimizer, loss_function],feed_dict={x: batch_x, y : batch_y, y_steps: batch_y_steps})   
        pred_y = session.run(y_last,feed_dict={x:test_x})
        print(" Training Accuracy: ", session.run(accuracy, feed_dict={x: train_x, y: train_y}))
        #print("ROC AUC Score: ",roc_auc_score(test_y,pred_y))


