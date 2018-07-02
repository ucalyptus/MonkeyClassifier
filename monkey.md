

```python
from __future__ import absolute_import,division,print_function
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from sklearn.utils import shuffle

```


```python
def load_data(path):
    folders = os.listdir(path)
    img_data = []
    label_data = []
    c = 0
    for folder in  folders:
        images=os.listdir(path+folder)
        for image in images:
            img = cv2.imread(path+folder+'/'+image)
            
            img = cv2.resize(img , (32,32) ,interpolation=cv2.INTER_AREA)

            label= folder.split("n")[1]
            
            print(img_data.append(img))

            label_data.append(label)

            c = c + 1


    img_data = np.array(img_data)


    label_data=to_myone_hot(label_data)
    label_data=np.argmax(label_data,1)

    img_data, label_data = shuffle(img_data, label_data, random_state=0)
    return img_data, label_data 


```


```python
def to_myone_hot(label_data):

    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import OneHotEncoder
    from numpy import array
    values=array(label_data)

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded =  integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    
    return onehot_encoded
```


```python
def Network(Input): #input : [Batch_size, 32, 32, 1]
    with tf.name_scope("Network"):
       
        conv1_1 = tf.layers.conv2d(Input, filters = 64, kernel_size = 3, strides = 1, padding='SAME', activation = tf.nn.relu, name = 'conv1_1')
        conv1_2 = tf.layers.conv2d(conv1_1, filters = 64, kernel_size = 3, strides = 1, padding='SAME', activation = tf.nn.relu, name = 'conv1_2')
        pool1 = tf.layers.max_pooling2d(conv1_2, pool_size = 2, strides = 2, padding='SAME', name = 'pool1')
        
        conv2_1 = tf.layers.conv2d(pool1, filters = 128, kernel_size = 3, strides = 1, padding='SAME', activation = tf.nn.relu, name = 'conv2_1')
        conv2_2 = tf.layers.conv2d(conv2_1, filters = 128, kernel_size = 3, strides = 1, padding='SAME', activation = tf.nn.relu, name = 'conv2_2')
        pool2 = tf.layers.max_pooling2d(conv2_2, pool_size = 2, strides = 2, padding='SAME', name = 'pool2')

        conv3_1 = tf.layers.conv2d(pool2, filters = 256, kernel_size = 3, strides = 1, padding='SAME', activation = tf.nn.relu, name = 'conv3_1')
        conv3_2 = tf.layers.conv2d(conv3_1, filters = 256, kernel_size = 3, strides = 1, padding='SAME', activation = tf.nn.relu, name = 'conv3_2')
        conv3_3 = tf.layers.conv2d(conv3_2, filters = 256, kernel_size = 3, strides = 1, padding='SAME', activation = tf.nn.relu, name = 'conv3_3')
        pool3 = tf.layers.max_pooling2d(conv3_3, pool_size = 2, strides = 2, padding='SAME', name = 'pool3')
        
        conv4_1 = tf.layers.conv2d(pool3, filters = 512, kernel_size = 3, strides = 1, padding='SAME', activation = tf.nn.relu, name = 'conv4_1')
        conv4_2 = tf.layers.conv2d(conv4_1, filters = 512, kernel_size = 3, strides = 1, padding='SAME', activation = tf.nn.relu, name = 'conv4_2')
        conv4_3 = tf.layers.conv2d(conv4_2, filters = 512, kernel_size = 3, strides = 1, padding='SAME', activation = tf.nn.relu, name = 'conv4_3')
        pool4 = tf.layers.max_pooling2d(conv4_3, pool_size = 2, strides = 2, padding='SAME', name = 'pool4')
        
        conv5_1 = tf.layers.conv2d(pool4, filters = 512, kernel_size = 3, strides = 1, padding='SAME', activation = tf.nn.relu, name = 'conv5_1')
        conv5_2 = tf.layers.conv2d(conv5_1, filters = 512, kernel_size = 3, strides = 1, padding='SAME', activation = tf.nn.relu, name = 'conv5_2')
        conv5_3 = tf.layers.conv2d(conv5_2, filters = 512, kernel_size = 3, strides = 1, padding='SAME', activation = tf.nn.relu, name = 'conv5_3')
    
        
        flat = tf.contrib.layers.flatten(conv5_3)
        fc1 = tf.layers.dense(flat, units = 1024, activation = tf.nn.relu, name = 'fc1')
        fc2 = tf.layers.dense(fc1, units = 256, activation = None, name = 'fc2')
        fc3 = tf.layers.dense(fc2, units = 10, activation = None, name = 'fc3')
        
     
    return fc3
    
    
```


```python
def initialize_vgg():
    weight_file = 'vgg16_weights.npz'
    weights = np.load(weight_file)
    keys = sorted(weights.keys())
    keys = keys[:-6]
    for i, k in enumerate(keys):
        #print(i, k, np.shape(weights[k]), tf.trainable_variables()[i])
        sess.run(tf.trainable_variables()[i].assign(weights[k]))
    print('Loaded from VGG')
```


```python
def loss_function(logit, Label):
    
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logit, labels = Label))
    
    return loss
```


```python
def Accuracy_Evaluate(prediction, Label):
    # Evaluate model
    #correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Label, 1))
    correct_pred = tf.equal(tf.argmax(tf.nn.softmax(prediction), 1), tf.cast(Label, tf.int64))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return correct_pred, accuracy
```


```python
def main(train_data, train_label, no_of_epochs = 150000, batchsize = 32):
    
    
    Input = tf.placeholder(dtype = tf.float32, shape = [batchsize,32,32,3])
    Label = tf.placeholder(dtype = tf.int32, shape = [batchsize])
    
    
    logit = Network(Input)
    
    
    prediction = tf.nn.softmax(logit)
    
    loss = loss_function(logit, Label)
    
    correct_pred, accuracy = Accuracy_Evaluate(prediction, Label)
    
    
    optimiz = tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(loss)

    init = tf.global_variables_initializer()
    sess.run(init)
    
    initialize_vgg()
 
    
    tf.summary.scalar('Loss_Value',loss)
    tf.summary.scalar('Accuracy',accuracy)
    
    print('Setting up summary op...')
    summary_op = tf.summary.merge_all()
    
    print('Setting Up Saver...')
    summary_writer = tf.summary.FileWriter('./log_dir/', sess.graph)
    
    itr = 0
    for epoch in range(no_of_epochs):
        
        index = np.random.permutation(np.shape(train_data)[0])
        train_data = train_data[index, :, :, :]
        train_label = train_label[index]
        
        for idx in range(train_data.shape[0]//batchsize): 
            
            batchx = train_data[idx*batchsize : (idx + 1)*batchsize , :, :, :]
            batchy = train_label[idx*batchsize : (idx + 1)*batchsize]
            
            feed_dict = {Input : batchx , Label : batchy}
            
            _, train_loss, train_accuracy, summary_str = sess.run([optimiz, loss, accuracy, summary_op] , feed_dict )
            summary_writer.add_summary(summary_str, itr)
            itr = itr + 1
            
            if idx%10 == 0:
                
                print ('epoch : '+str(epoch)+' step : '+str(idx) + ' train_loss : '+str(train_loss) +
                        ' train_accuracy : '+str(train_accuracy) 
            
                      )
```


```python
from tensorflow.python.framework import ops
ops.reset_default_graph()
global sess

config = tf.ConfigProto()
sess = tf.Session(config = config)
graph = tf.get_default_graph()
```


```python
train_data, train_label =load_data('/root/Desktop/monkey/training/')
```

    (1358, 2048, 3)
    (32, 32, 3)
    None
    (1166, 1702, 3)
    (32, 32, 3)
    None
    (466, 639, 3)
    (32, 32, 3)
    None
    (900, 607, 3)
    (32, 32, 3)
    None
    (322, 484, 3)
    (32, 32, 3)
    None
    (494, 655, 3)
    (32, 32, 3)
    None
    (742, 989, 3)
    (32, 32, 3)
    None
    (866, 1300, 3)
    (32, 32, 3)
    None
    (1358, 2048, 3)
    (32, 32, 3)
    None
    (396, 594, 3)
    (32, 32, 3)
    None
    (629, 944, 3)
    (32, 32, 3)
    None
    (1280, 2048, 3)
    (32, 32, 3)
    None
    (200, 300, 3)
    (32, 32, 3)
    None
    (480, 852, 3)
    (32, 32, 3)
    None
    (367, 550, 3)
    (32, 32, 3)
    None
    (1365, 2048, 3)
    (32, 32, 3)
    None
    (743, 990, 3)
    (32, 32, 3)
    None
    (999, 1500, 3)
    (32, 32, 3)
    None
    (447, 640, 3)
    (32, 32, 3)
    None
    (608, 448, 3)
    (32, 32, 3)
    None
    (666, 1000, 3)
    (32, 32, 3)
    None
    (370, 470, 3)
    (32, 32, 3)
    None
    (863, 1296, 3)
    (32, 32, 3)
    None
    (1080, 1920, 3)
    (32, 32, 3)
    None
    (720, 540, 3)
    (32, 32, 3)
    None
    (533, 800, 3)
    (32, 32, 3)
    None
    (861, 1293, 3)
    (32, 32, 3)
    None
    (447, 500, 3)
    (32, 32, 3)
    None
    (480, 852, 3)
    (32, 32, 3)
    None
    (527, 360, 3)
    (32, 32, 3)
    None
    (2592, 3888, 3)
    (32, 32, 3)
    None
    (370, 470, 3)
    (32, 32, 3)
    None
    (533, 800, 3)
    (32, 32, 3)
    None
    (500, 332, 3)
    (32, 32, 3)
    None
    (433, 650, 3)
    (32, 32, 3)
    None
    (865, 1300, 3)
    (32, 32, 3)
    None
    (536, 944, 3)
    (32, 32, 3)
    None
    (409, 615, 3)
    (32, 32, 3)
    None
    (533, 800, 3)
    (32, 32, 3)
    None
    (333, 500, 3)
    (32, 32, 3)
    None
    (435, 620, 3)
    (32, 32, 3)
    None
    (641, 641, 3)
    (32, 32, 3)
    None
    (500, 378, 3)
    (32, 32, 3)
    None
    (1358, 2048, 3)
    (32, 32, 3)
    None
    (604, 750, 3)
    (32, 32, 3)
    None
    (667, 1000, 3)
    (32, 32, 3)
    None
    (600, 600, 3)
    (32, 32, 3)
    None
    (325, 402, 3)
    (32, 32, 3)
    None
    (1293, 905, 3)
    (32, 32, 3)
    None
    (560, 1120, 3)
    (32, 32, 3)
    None
    (821, 1295, 3)
    (32, 32, 3)
    None
    (999, 1500, 3)
    (32, 32, 3)
    None
    (865, 1297, 3)
    (32, 32, 3)
    None
    (490, 944, 3)
    (32, 32, 3)
    None
    (1995, 1600, 3)
    (32, 32, 3)
    None
    (350, 600, 3)
    (32, 32, 3)
    None
    (500, 800, 3)
    (32, 32, 3)
    None
    (599, 401, 3)
    (32, 32, 3)
    None
    (534, 800, 3)
    (32, 32, 3)
    None
    (1000, 1500, 3)
    (32, 32, 3)
    None
    (335, 500, 3)
    (32, 32, 3)
    None
    (1365, 2048, 3)
    (32, 32, 3)
    None
    (1600, 2560, 3)
    (32, 32, 3)
    None
    (996, 1494, 3)
    (32, 32, 3)
    None
    (500, 438, 3)
    (32, 32, 3)
    None
    (498, 750, 3)
    (32, 32, 3)
    None
    (1325, 1995, 3)
    (32, 32, 3)
    None
    (865, 1300, 3)
    (32, 32, 3)
    None
    (480, 614, 3)
    (32, 32, 3)
    None
    (334, 500, 3)
    (32, 32, 3)
    None
    (866, 1300, 3)
    (32, 32, 3)
    None
    (750, 498, 3)
    (32, 32, 3)
    None
    (533, 800, 3)
    (32, 32, 3)
    None
    (1272, 1007, 3)
    (32, 32, 3)
    None
    (733, 1100, 3)
    (32, 32, 3)
    None
    (453, 680, 3)
    (32, 32, 3)
    None
    (1283, 860, 3)
    (32, 32, 3)
    None
    (1536, 1920, 3)
    (32, 32, 3)
    None
    (404, 600, 3)
    (32, 32, 3)
    None
    (863, 1298, 3)
    (32, 32, 3)
    None
    (1280, 1920, 3)
    (32, 32, 3)
    None
    (1600, 1200, 3)
    (32, 32, 3)
    None
    (480, 614, 3)
    (32, 32, 3)
    None
    (378, 564, 3)
    (32, 32, 3)
    None
    (1280, 1600, 3)
    (32, 32, 3)
    None
    (525, 700, 3)
    (32, 32, 3)
    None
    (472, 674, 3)
    (32, 32, 3)
    None
    (335, 500, 3)
    (32, 32, 3)
    None
    (600, 400, 3)
    (32, 32, 3)
    None
    (312, 394, 3)
    (32, 32, 3)
    None
    (1299, 885, 3)
    (32, 32, 3)
    None
    (682, 1023, 3)
    (32, 32, 3)
    None
    (399, 600, 3)
    (32, 32, 3)
    None
    (430, 615, 3)
    (32, 32, 3)
    None
    (1291, 904, 3)
    (32, 32, 3)
    None
    (198, 384, 3)
    (32, 32, 3)
    None
    (2848, 4288, 3)
    (32, 32, 3)
    None
    (1103, 736, 3)
    (32, 32, 3)
    None
    (437, 650, 3)
    (32, 32, 3)
    None
    (290, 450, 3)
    (32, 32, 3)
    None
    (630, 1200, 3)
    (32, 32, 3)
    None
    (500, 800, 3)
    (32, 32, 3)
    None
    (375, 500, 3)
    (32, 32, 3)
    None
    (556, 659, 3)
    (32, 32, 3)
    None
    (3168, 4752, 3)
    (32, 32, 3)
    None
    (597, 900, 3)
    (32, 32, 3)
    None
    (866, 1300, 3)
    (32, 32, 3)
    None
    (720, 480, 3)
    (32, 32, 3)
    None
    (400, 300, 3)
    (32, 32, 3)
    None
    (574, 750, 3)
    (32, 32, 3)
    None
    (401, 600, 3)
    (32, 32, 3)
    None
    (467, 700, 3)
    (32, 32, 3)
    None
    (1490, 2000, 3)
    (32, 32, 3)
    None
    (700, 525, 3)
    (32, 32, 3)
    None
    (680, 634, 3)
    (32, 32, 3)
    None
    (400, 640, 3)
    (32, 32, 3)
    None
    (1424, 1900, 3)
    (32, 32, 3)
    None
    (932, 1442, 3)
    (32, 32, 3)
    None
    (562, 944, 3)
    (32, 32, 3)
    None
    (600, 600, 3)
    (32, 32, 3)
    None
    (534, 800, 3)
    (32, 32, 3)
    None
    (771, 1235, 3)
    (32, 32, 3)
    None
    (300, 450, 3)
    (32, 32, 3)
    None
    (768, 1024, 3)
    (32, 32, 3)
    None
    (683, 1024, 3)
    (32, 32, 3)
    None
    (501, 419, 3)
    (32, 32, 3)
    None
    (658, 800, 3)
    (32, 32, 3)
    None
    (800, 500, 3)
    (32, 32, 3)
    None
    (487, 650, 3)
    (32, 32, 3)
    None
    (525, 700, 3)
    (32, 32, 3)
    None
    (864, 1300, 3)
    (32, 32, 3)
    None
    (424, 500, 3)
    (32, 32, 3)
    None
    (429, 506, 3)
    (32, 32, 3)
    None
    (975, 1300, 3)
    (32, 32, 3)
    None
    (530, 750, 3)
    (32, 32, 3)
    None
    (800, 800, 3)
    (32, 32, 3)
    None
    (1300, 1296, 3)
    (32, 32, 3)
    None
    (811, 1024, 3)
    (32, 32, 3)
    None
    (299, 450, 3)
    (32, 32, 3)
    None
    (768, 1024, 3)
    (32, 32, 3)
    None
    (526, 736, 3)
    (32, 32, 3)
    None
    (683, 1024, 3)
    (32, 32, 3)
    None
    (435, 341, 3)
    (32, 32, 3)
    None
    (473, 428, 3)
    (32, 32, 3)
    None
    (600, 510, 3)
    (32, 32, 3)
    None
    (678, 800, 3)
    (32, 32, 3)
    None
    (400, 400, 3)
    (32, 32, 3)
    None
    (630, 750, 3)
    (32, 32, 3)
    None
    (4272, 2848, 3)
    (32, 32, 3)
    None
    (600, 800, 3)
    (32, 32, 3)
    None
    (525, 700, 3)
    (32, 32, 3)
    None
    (598, 690, 3)
    (32, 32, 3)
    None
    (600, 600, 3)
    (32, 32, 3)
    None
    (864, 1300, 3)
    (32, 32, 3)
    None
    (448, 300, 3)
    (32, 32, 3)
    None
    (914, 1300, 3)
    (32, 32, 3)
    None
    (433, 398, 3)
    (32, 32, 3)
    None
    (998, 1500, 3)
    (32, 32, 3)
    None
    (928, 1300, 3)
    (32, 32, 3)
    None
    (866, 1300, 3)
    (32, 32, 3)
    None
    (980, 1300, 3)
    (32, 32, 3)
    None
    (481, 800, 3)
    (32, 32, 3)
    None
    (866, 1300, 3)
    (32, 32, 3)
    None
    (479, 800, 3)
    (32, 32, 3)
    None
    (426, 640, 3)
    (32, 32, 3)
    None
    (487, 650, 3)
    (32, 32, 3)
    None
    (425, 640, 3)
    (32, 32, 3)
    None
    (2348, 3522, 3)
    (32, 32, 3)
    None
    (718, 519, 3)
    (32, 32, 3)
    None
    (447, 364, 3)
    (32, 32, 3)
    None
    (864, 1300, 3)
    (32, 32, 3)
    None
    (480, 852, 3)
    (32, 32, 3)
    None
    (500, 500, 3)
    (32, 32, 3)
    None
    (3408, 2348, 3)
    (32, 32, 3)
    None
    (313, 600, 3)
    (32, 32, 3)
    None
    (591, 600, 3)
    (32, 32, 3)
    None
    (706, 1060, 3)
    (32, 32, 3)
    None
    (644, 1024, 3)
    (32, 32, 3)
    None
    (1300, 866, 3)
    (32, 32, 3)
    None
    (424, 640, 3)
    (32, 32, 3)
    None
    (1000, 1300, 3)
    (32, 32, 3)
    None
    (684, 1024, 3)
    (32, 32, 3)
    None
    (400, 400, 3)
    (32, 32, 3)
    None
    (439, 600, 3)
    (32, 32, 3)
    None
    (450, 300, 3)
    (32, 32, 3)
    None
    (499, 800, 3)
    (32, 32, 3)
    None
    (1301, 866, 3)
    (32, 32, 3)
    None
    (408, 650, 3)
    (32, 32, 3)
    None
    (375, 500, 3)
    (32, 32, 3)
    None
    (865, 1300, 3)
    (32, 32, 3)
    None
    (869, 1200, 3)
    (32, 32, 3)
    None
    (950, 1300, 3)
    (32, 32, 3)
    None
    (768, 1024, 3)
    (32, 32, 3)
    None
    (598, 800, 3)
    (32, 32, 3)
    None
    (450, 338, 3)
    (32, 32, 3)
    None
    (480, 960, 3)
    (32, 32, 3)
    None
    (854, 1300, 3)
    (32, 32, 3)
    None
    (404, 650, 3)
    (32, 32, 3)
    None
    (546, 750, 3)
    (32, 32, 3)
    None
    (2588, 1725, 3)
    (32, 32, 3)
    None
    (863, 1300, 3)
    (32, 32, 3)
    None
    (498, 750, 3)
    (32, 32, 3)
    None
    (450, 450, 3)
    (32, 32, 3)
    None
    (380, 530, 3)
    (32, 32, 3)
    None
    (1001, 1500, 3)
    (32, 32, 3)
    None
    (487, 650, 3)
    (32, 32, 3)
    None
    (525, 700, 3)
    (32, 32, 3)
    None
    (517, 347, 3)
    (32, 32, 3)
    None
    (336, 427, 3)
    (32, 32, 3)
    None
    (692, 736, 3)
    (32, 32, 3)
    None
    (532, 800, 3)
    (32, 32, 3)
    None
    (199, 351, 3)
    (32, 32, 3)
    None
    (682, 1024, 3)
    (32, 32, 3)
    None
    (993, 1500, 3)
    (32, 32, 3)
    None
    (2912, 4368, 3)
    (32, 32, 3)
    None
    (541, 750, 3)
    (32, 32, 3)
    None
    (433, 338, 3)
    (32, 32, 3)
    None
    (473, 800, 3)
    (32, 32, 3)
    None
    (1300, 866, 3)
    (32, 32, 3)
    None
    (866, 1300, 3)
    (32, 32, 3)
    None
    (1298, 866, 3)
    (32, 32, 3)
    None
    (687, 750, 3)
    (32, 32, 3)
    None
    (550, 700, 3)
    (32, 32, 3)
    None
    (338, 450, 3)
    (32, 32, 3)
    None
    (500, 500, 3)
    (32, 32, 3)
    None
    (869, 1200, 3)
    (32, 32, 3)
    None
    (730, 1095, 3)
    (32, 32, 3)
    None
    (428, 750, 3)
    (32, 32, 3)
    None
    (900, 1200, 3)
    (32, 32, 3)
    None
    (473, 473, 3)
    (32, 32, 3)
    None
    (448, 450, 3)
    (32, 32, 3)
    None
    (433, 650, 3)
    (32, 32, 3)
    None
    (350, 400, 3)
    (32, 32, 3)
    None
    (300, 450, 3)
    (32, 32, 3)
    None
    (400, 400, 3)
    (32, 32, 3)
    None
    (292, 438, 3)
    (32, 32, 3)
    None
    (661, 800, 3)
    (32, 32, 3)
    None
    (433, 649, 3)
    (32, 32, 3)
    None
    (1200, 858, 3)
    (32, 32, 3)
    None
    (1223, 1200, 3)
    (32, 32, 3)
    None
    (900, 1200, 3)
    (32, 32, 3)
    None
    (374, 704, 3)
    (32, 32, 3)
    None
    (600, 760, 3)
    (32, 32, 3)
    None
    (941, 1300, 3)
    (32, 32, 3)
    None
    (520, 780, 3)
    (32, 32, 3)
    None
    (333, 500, 3)
    (32, 32, 3)
    None
    (800, 1200, 3)
    (32, 32, 3)
    None
    (465, 700, 3)
    (32, 32, 3)
    None
    (1920, 2560, 3)
    (32, 32, 3)
    None
    (465, 700, 3)
    (32, 32, 3)
    None
    (599, 546, 3)
    (32, 32, 3)
    None
    (931, 1200, 3)
    (32, 32, 3)
    None
    (465, 700, 3)
    (32, 32, 3)
    None
    (1275, 861, 3)
    (32, 32, 3)
    None
    (374, 704, 3)
    (32, 32, 3)
    None
    (1784, 2400, 3)
    (32, 32, 3)
    None
    (295, 448, 3)
    (32, 32, 3)
    None
    (447, 591, 3)
    (32, 32, 3)
    None
    (433, 650, 3)
    (32, 32, 3)
    None
    (800, 533, 3)
    (32, 32, 3)
    None
    (640, 1280, 3)
    (32, 32, 3)
    None
    (443, 295, 3)
    (32, 32, 3)
    None
    (468, 700, 3)
    (32, 32, 3)
    None
    (435, 650, 3)
    (32, 32, 3)
    None
    (1118, 1491, 3)
    (32, 32, 3)
    None
    (285, 426, 3)
    (32, 32, 3)
    None
    (600, 800, 3)
    (32, 32, 3)
    None
    (834, 1293, 3)
    (32, 32, 3)
    None
    (895, 1180, 3)
    (32, 32, 3)
    None
    (374, 704, 3)
    (32, 32, 3)
    None
    (295, 562, 3)
    (32, 32, 3)
    None
    (997, 1493, 3)
    (32, 32, 3)
    None
    (468, 700, 3)
    (32, 32, 3)
    None
    (863, 1295, 3)
    (32, 32, 3)
    None
    (600, 800, 3)
    (32, 32, 3)
    None
    (443, 665, 3)
    (32, 32, 3)
    None
    (600, 600, 3)
    (32, 32, 3)
    None
    (400, 600, 3)
    (32, 32, 3)
    None
    (733, 1100, 3)
    (32, 32, 3)
    None
    (500, 800, 3)
    (32, 32, 3)
    None
    (762, 1024, 3)
    (32, 32, 3)
    None
    (295, 446, 3)
    (32, 32, 3)
    None
    (570, 857, 3)
    (32, 32, 3)
    None
    (532, 800, 3)
    (32, 32, 3)
    None
    (599, 900, 3)
    (32, 32, 3)
    None
    (700, 460, 3)
    (32, 32, 3)
    None
    (333, 500, 3)
    (32, 32, 3)
    None
    (422, 633, 3)
    (32, 32, 3)
    None
    (374, 704, 3)
    (32, 32, 3)
    None
    (861, 1297, 3)
    (32, 32, 3)
    None
    (866, 1300, 3)
    (32, 32, 3)
    None
    (500, 800, 3)
    (32, 32, 3)
    None
    (596, 800, 3)
    (32, 32, 3)
    None
    (520, 767, 3)
    (32, 32, 3)
    None
    (1123, 1447, 3)
    (32, 32, 3)
    None
    (800, 1200, 3)
    (32, 32, 3)
    None
    (700, 466, 3)
    (32, 32, 3)
    None
    (1090, 1300, 3)
    (32, 32, 3)
    None
    (640, 1280, 3)
    (32, 32, 3)
    None
    (2112, 2358, 3)
    (32, 32, 3)
    None
    (461, 700, 3)
    (32, 32, 3)
    None
    (683, 1024, 3)
    (32, 32, 3)
    None
    (763, 798, 3)
    (32, 32, 3)
    None
    (678, 1024, 3)
    (32, 32, 3)
    None
    (900, 1200, 3)
    (32, 32, 3)
    None
    (960, 1280, 3)
    (32, 32, 3)
    None
    (800, 1200, 3)
    (32, 32, 3)
    None
    (458, 610, 3)
    (32, 32, 3)
    None
    (2832, 4256, 3)
    (32, 32, 3)
    None
    (680, 1024, 3)
    (32, 32, 3)
    None
    (900, 600, 3)
    (32, 32, 3)
    None
    (374, 704, 3)
    (32, 32, 3)
    None
    (800, 500, 3)
    (32, 32, 3)
    None
    (285, 426, 3)
    (32, 32, 3)
    None
    (427, 640, 3)
    (32, 32, 3)
    None
    (836, 1296, 3)
    (32, 32, 3)
    None
    (525, 700, 3)
    (32, 32, 3)
    None
    (480, 491, 3)
    (32, 32, 3)
    None
    (590, 800, 3)
    (32, 32, 3)
    None
    (866, 1300, 3)
    (32, 32, 3)
    None
    (800, 1200, 3)
    (32, 32, 3)
    None
    (600, 800, 3)
    (32, 32, 3)
    None
    (856, 1293, 3)
    (32, 32, 3)
    None
    (450, 600, 3)
    (32, 32, 3)
    None
    (720, 1280, 3)
    (32, 32, 3)
    None
    (960, 1280, 3)
    (32, 32, 3)
    None
    (370, 470, 3)
    (32, 32, 3)
    None
    (866, 1300, 3)
    (32, 32, 3)
    None
    (640, 1280, 3)
    (32, 32, 3)
    None
    (520, 780, 3)
    (32, 32, 3)
    None
    (857, 1200, 3)
    (32, 32, 3)
    None
    (1470, 1960, 3)
    (32, 32, 3)
    None
    (480, 640, 3)
    (32, 32, 3)
    None
    (230, 270, 3)
    (32, 32, 3)
    None
    (520, 780, 3)
    (32, 32, 3)
    None
    (900, 594, 3)
    (32, 32, 3)
    None
    (881, 1280, 3)
    (32, 32, 3)
    None
    (295, 562, 3)
    (32, 32, 3)
    None
    (374, 704, 3)
    (32, 32, 3)
    None
    (600, 1280, 3)
    (32, 32, 3)
    None
    (640, 1280, 3)
    (32, 32, 3)
    None
    (895, 866, 3)
    (32, 32, 3)
    None
    (473, 700, 3)
    (32, 32, 3)
    None
    (433, 650, 3)
    (32, 32, 3)
    None
    (1033, 857, 3)
    (32, 32, 3)
    None
    (929, 1280, 3)
    (32, 32, 3)
    None
    (300, 450, 3)
    (32, 32, 3)
    None
    (433, 650, 3)
    (32, 32, 3)
    None
    (1364, 2048, 3)
    (32, 32, 3)
    None
    (840, 600, 3)
    (32, 32, 3)
    None
    (1024, 854, 3)
    (32, 32, 3)
    None
    (683, 1024, 3)
    (32, 32, 3)
    None
    (533, 800, 3)
    (32, 32, 3)
    None
    (300, 450, 3)
    (32, 32, 3)
    None
    (296, 450, 3)
    (32, 32, 3)
    None
    (1600, 1065, 3)
    (32, 32, 3)
    None
    (500, 392, 3)
    (32, 32, 3)
    None
    (1336, 891, 3)
    (32, 32, 3)
    None
    (866, 1300, 3)
    (32, 32, 3)
    None
    (520, 347, 3)
    (32, 32, 3)
    None
    (366, 550, 3)
    (32, 32, 3)
    None
    (293, 428, 3)
    (32, 32, 3)
    None
    (682, 1024, 3)
    (32, 32, 3)
    None
    (768, 1024, 3)
    (32, 32, 3)
    None
    (2691, 2018, 3)
    (32, 32, 3)
    None
    (731, 1024, 3)
    (32, 32, 3)
    None
    (414, 537, 3)
    (32, 32, 3)
    None
    (300, 450, 3)
    (32, 32, 3)
    None
    (600, 1024, 3)
    (32, 32, 3)
    None
    (450, 300, 3)
    (32, 32, 3)
    None
    (900, 1200, 3)
    (32, 32, 3)
    None
    (853, 1280, 3)
    (32, 32, 3)
    None
    (432, 650, 3)
    (32, 32, 3)
    None
    (332, 500, 3)
    (32, 32, 3)
    None
    (533, 800, 3)
    (32, 32, 3)
    None
    (331, 500, 3)
    (32, 32, 3)
    None
    (466, 700, 3)
    (32, 32, 3)
    None
    (709, 1024, 3)
    (32, 32, 3)
    None
    (433, 650, 3)
    (32, 32, 3)
    None
    (432, 650, 3)
    (32, 32, 3)
    None
    (679, 1024, 3)
    (32, 32, 3)
    None
    (331, 500, 3)
    (32, 32, 3)
    None
    (720, 1280, 3)
    (32, 32, 3)
    None
    (852, 1296, 3)
    (32, 32, 3)
    None
    (409, 297, 3)
    (32, 32, 3)
    None
    (450, 570, 3)
    (32, 32, 3)
    None
    (768, 1024, 3)
    (32, 32, 3)
    None
    (366, 550, 3)
    (32, 32, 3)
    None
    (331, 500, 3)
    (32, 32, 3)
    None
    (1080, 1920, 3)
    (32, 32, 3)
    None
    (763, 1000, 3)
    (32, 32, 3)
    None
    (576, 1024, 3)
    (32, 32, 3)
    None
    (338, 450, 3)
    (32, 32, 3)
    None
    (448, 304, 3)
    (32, 32, 3)
    None
    (1336, 891, 3)
    (32, 32, 3)
    None
    (331, 500, 3)
    (32, 32, 3)
    None
    (367, 550, 3)
    (32, 32, 3)
    None
    (426, 640, 3)
    (32, 32, 3)
    None
    (768, 1080, 3)
    (32, 32, 3)
    None
    (1024, 1024, 3)
    (32, 32, 3)
    None
    (331, 500, 3)
    (32, 32, 3)
    None
    (450, 337, 3)
    (32, 32, 3)
    None
    (447, 650, 3)
    (32, 32, 3)
    None
    (433, 650, 3)
    (32, 32, 3)
    None
    (500, 800, 3)
    (32, 32, 3)
    None
    (466, 700, 3)
    (32, 32, 3)
    None
    (520, 358, 3)
    (32, 32, 3)
    None
    (545, 1024, 3)
    (32, 32, 3)
    None
    (447, 650, 3)
    (32, 32, 3)
    None
    (1200, 1900, 3)
    (32, 32, 3)
    None
    (683, 1024, 3)
    (32, 32, 3)
    None
    (275, 183, 3)
    (32, 32, 3)
    None
    (433, 650, 3)
    (32, 32, 3)
    None
    (1062, 865, 3)
    (32, 32, 3)
    None
    (3456, 5184, 3)
    (32, 32, 3)
    None
    (600, 600, 3)
    (32, 32, 3)
    None
    (891, 1336, 3)
    (32, 32, 3)
    None
    (1900, 1526, 3)
    (32, 32, 3)
    None
    (705, 1063, 3)
    (32, 32, 3)
    None
    (2911, 4367, 3)
    (32, 32, 3)
    None
    (817, 896, 3)
    (32, 32, 3)
    None
    (568, 800, 3)
    (32, 32, 3)
    None
    (367, 550, 3)
    (32, 32, 3)
    None
    (3456, 4608, 3)
    (32, 32, 3)
    None
    (367, 550, 3)
    (32, 32, 3)
    None
    (433, 650, 3)
    (32, 32, 3)
    None
    (866, 1300, 3)
    (32, 32, 3)
    None
    (400, 600, 3)
    (32, 32, 3)
    None
    (679, 1024, 3)
    (32, 32, 3)
    None
    (1336, 891, 3)
    (32, 32, 3)
    None
    (897, 866, 3)
    (32, 32, 3)
    None
    (500, 800, 3)
    (32, 32, 3)
    None
    (266, 399, 3)
    (32, 32, 3)
    None
    (466, 700, 3)
    (32, 32, 3)
    None
    (333, 500, 3)
    (32, 32, 3)
    None
    (891, 1336, 3)
    (32, 32, 3)
    None
    (768, 1024, 3)
    (32, 32, 3)
    None
    (1300, 866, 3)
    (32, 32, 3)
    None
    (480, 852, 3)
    (32, 32, 3)
    None
    (367, 550, 3)
    (32, 32, 3)
    None
    (1069, 866, 3)
    (32, 32, 3)
    None
    (1251, 830, 3)
    (32, 32, 3)
    None
    (432, 650, 3)
    (32, 32, 3)
    None
    (432, 650, 3)
    (32, 32, 3)
    None
    (751, 1044, 3)
    (32, 32, 3)
    None
    (300, 450, 3)
    (32, 32, 3)
    None
    (300, 450, 3)
    (32, 32, 3)
    None
    (558, 800, 3)
    (32, 32, 3)
    None
    (480, 640, 3)
    (32, 32, 3)
    None
    (1301, 865, 3)
    (32, 32, 3)
    None
    (1300, 867, 3)
    (32, 32, 3)
    None
    (533, 800, 3)
    (32, 32, 3)
    None
    (468, 700, 3)
    (32, 32, 3)
    None
    (450, 300, 3)
    (32, 32, 3)
    None
    (866, 1300, 3)
    (32, 32, 3)
    None
    (600, 402, 3)
    (32, 32, 3)
    None
    (900, 600, 3)
    (32, 32, 3)
    None
    (520, 343, 3)
    (32, 32, 3)
    None
    (2112, 3170, 3)
    (32, 32, 3)
    None
    (970, 1234, 3)
    (32, 32, 3)
    None
    (336, 502, 3)
    (32, 32, 3)
    None
    (435, 650, 3)
    (32, 32, 3)
    None
    (520, 348, 3)
    (32, 32, 3)
    None
    (450, 300, 3)
    (32, 32, 3)
    None
    (366, 550, 3)
    (32, 32, 3)
    None
    (867, 1300, 3)
    (32, 32, 3)
    None
    (430, 650, 3)
    (32, 32, 3)
    None
    (421, 800, 3)
    (32, 32, 3)
    None
    (2797, 4204, 3)
    (32, 32, 3)
    None
    (468, 700, 3)
    (32, 32, 3)
    None
    (683, 1024, 3)
    (32, 32, 3)
    None
    (299, 450, 3)
    (32, 32, 3)
    None
    (1302, 868, 3)
    (32, 32, 3)
    None
    (433, 650, 3)
    (32, 32, 3)
    None
    (334, 500, 3)
    (32, 32, 3)
    None
    (520, 347, 3)
    (32, 32, 3)
    None
    (700, 467, 3)
    (32, 32, 3)
    None
    (480, 640, 3)
    (32, 32, 3)
    None
    (866, 1300, 3)
    (32, 32, 3)
    None
    (797, 1200, 3)
    (32, 32, 3)
    None
    (435, 650, 3)
    (32, 32, 3)
    None
    (435, 650, 3)
    (32, 32, 3)
    None
    (435, 650, 3)
    (32, 32, 3)
    None
    (300, 450, 3)
    (32, 32, 3)
    None
    (683, 1023, 3)
    (32, 32, 3)
    None
    (300, 450, 3)
    (32, 32, 3)
    None
    (1302, 849, 3)
    (32, 32, 3)
    None
    (700, 468, 3)
    (32, 32, 3)
    None
    (1300, 866, 3)
    (32, 32, 3)
    None
    (417, 650, 3)
    (32, 32, 3)
    None
    (1497, 1000, 3)
    (32, 32, 3)
    None
    (850, 1280, 3)
    (32, 32, 3)
    None
    (867, 1298, 3)
    (32, 32, 3)
    None
    (300, 450, 3)
    (32, 32, 3)
    None
    (433, 650, 3)
    (32, 32, 3)
    None
    (865, 1298, 3)
    (32, 32, 3)
    None
    (435, 650, 3)
    (32, 32, 3)
    None
    (551, 736, 3)
    (32, 32, 3)
    None
    (1300, 862, 3)
    (32, 32, 3)
    None
    (501, 750, 3)
    (32, 32, 3)
    None
    (1536, 2048, 3)
    (32, 32, 3)
    None
    (344, 500, 3)
    (32, 32, 3)
    None
    (900, 600, 3)
    (32, 32, 3)
    None
    (870, 1298, 3)
    (32, 32, 3)
    None
    (646, 455, 3)
    (32, 32, 3)
    None
    (4000, 6000, 3)
    (32, 32, 3)
    None
    (700, 468, 3)
    (32, 32, 3)
    None
    (435, 650, 3)
    (32, 32, 3)
    None
    (435, 650, 3)
    (32, 32, 3)
    None
    (900, 1600, 3)
    (32, 32, 3)
    None
    (867, 1299, 3)
    (32, 32, 3)
    None
    (602, 1024, 3)
    (32, 32, 3)
    None
    (520, 346, 3)
    (32, 32, 3)
    None
    (520, 464, 3)
    (32, 32, 3)
    None
    (1278, 1920, 3)
    (32, 32, 3)
    None
    (797, 1200, 3)
    (32, 32, 3)
    None
    (1024, 768, 3)
    (32, 32, 3)
    None
    (700, 467, 3)
    (32, 32, 3)
    None
    (620, 414, 3)
    (32, 32, 3)
    None
    (354, 400, 3)
    (32, 32, 3)
    None
    (432, 650, 3)
    (32, 32, 3)
    None
    (411, 507, 3)
    (32, 32, 3)
    None
    (1300, 866, 3)
    (32, 32, 3)
    None
    (800, 600, 3)
    (32, 32, 3)
    None
    (520, 399, 3)
    (32, 32, 3)
    None
    (869, 1298, 3)
    (32, 32, 3)
    None
    (680, 1024, 3)
    (32, 32, 3)
    None
    (336, 502, 3)
    (32, 32, 3)
    None
    (436, 650, 3)
    (32, 32, 3)
    None
    (1300, 851, 3)
    (32, 32, 3)
    None
    (777, 653, 3)
    (32, 32, 3)
    None
    (1600, 1239, 3)
    (32, 32, 3)
    None
    (520, 464, 3)
    (32, 32, 3)
    None
    (600, 900, 3)
    (32, 32, 3)
    None
    (417, 650, 3)
    (32, 32, 3)
    None
    (1302, 864, 3)
    (32, 32, 3)
    None
    (470, 646, 3)
    (32, 32, 3)
    None
    (436, 653, 3)
    (32, 32, 3)
    None
    (498, 664, 3)
    (32, 32, 3)
    None
    (488, 366, 3)
    (32, 32, 3)
    None
    (640, 428, 3)
    (32, 32, 3)
    None
    (1024, 683, 3)
    (32, 32, 3)
    None
    (354, 400, 3)
    (32, 32, 3)
    None
    (593, 788, 3)
    (32, 32, 3)
    None
    (767, 1100, 3)
    (32, 32, 3)
    None
    (520, 344, 3)
    (32, 32, 3)
    None
    (900, 600, 3)
    (32, 32, 3)
    None
    (400, 400, 3)
    (32, 32, 3)
    None
    (1411, 1993, 3)
    (32, 32, 3)
    None
    (866, 1299, 3)
    (32, 32, 3)
    None
    (1302, 862, 3)
    (32, 32, 3)
    None
    (450, 299, 3)
    (32, 32, 3)
    None
    (866, 1300, 3)
    (32, 32, 3)
    None
    (900, 600, 3)
    (32, 32, 3)
    None
    (1300, 866, 3)
    (32, 32, 3)
    None
    (800, 533, 3)
    (32, 32, 3)
    None
    (600, 600, 3)
    (32, 32, 3)
    None
    (978, 1440, 3)
    (32, 32, 3)
    None
    (1023, 1149, 3)
    (32, 32, 3)
    None
    (675, 1200, 3)
    (32, 32, 3)
    None
    (370, 470, 3)
    (32, 32, 3)
    None
    (550, 392, 3)
    (32, 32, 3)
    None
    (543, 800, 3)
    (32, 32, 3)
    None
    (973, 1300, 3)
    (32, 32, 3)
    None
    (450, 314, 3)
    (32, 32, 3)
    None
    (293, 362, 3)
    (32, 32, 3)
    None
    (533, 800, 3)
    (32, 32, 3)
    None
    (1236, 822, 3)
    (32, 32, 3)
    None
    (999, 750, 3)
    (32, 32, 3)
    None
    (848, 1200, 3)
    (32, 32, 3)
    None
    (610, 960, 3)
    (32, 32, 3)
    None
    (407, 550, 3)
    (32, 32, 3)
    None
    (654, 1020, 3)
    (32, 32, 3)
    None
    (598, 800, 3)
    (32, 32, 3)
    None
    (456, 800, 3)
    (32, 32, 3)
    None
    (525, 792, 3)
    (32, 32, 3)
    None
    (750, 673, 3)
    (32, 32, 3)
    None
    (520, 346, 3)
    (32, 32, 3)
    None
    (560, 800, 3)
    (32, 32, 3)
    None
    (299, 345, 3)
    (32, 32, 3)
    None
    (866, 1300, 3)
    (32, 32, 3)
    None
    (552, 736, 3)
    (32, 32, 3)
    None
    (374, 704, 3)
    (32, 32, 3)
    None
    (2304, 1728, 3)
    (32, 32, 3)
    None
    (541, 700, 3)
    (32, 32, 3)
    None
    (784, 960, 3)
    (32, 32, 3)
    None
    (717, 736, 3)
    (32, 32, 3)
    None
    (294, 441, 3)
    (32, 32, 3)
    None
    (818, 1024, 3)
    (32, 32, 3)
    None
    (432, 650, 3)
    (32, 32, 3)
    None
    (371, 371, 3)
    (32, 32, 3)
    None
    (451, 600, 3)
    (32, 32, 3)
    None
    (667, 1000, 3)
    (32, 32, 3)
    None
    (1200, 1600, 3)
    (32, 32, 3)
    None
    (512, 1280, 3)
    (32, 32, 3)
    None
    (621, 800, 3)
    (32, 32, 3)
    None
    (379, 400, 3)
    (32, 32, 3)
    None
    (400, 500, 3)
    (32, 32, 3)
    None
    (275, 394, 3)
    (32, 32, 3)
    None
    (984, 915, 3)
    (32, 32, 3)
    None
    (986, 960, 3)
    (32, 32, 3)
    None
    (598, 800, 3)
    (32, 32, 3)
    None
    (866, 1300, 3)
    (32, 32, 3)
    None
    (456, 666, 3)
    (32, 32, 3)
    None
    (533, 800, 3)
    (32, 32, 3)
    None
    (1035, 1413, 3)
    (32, 32, 3)
    None
    (420, 440, 3)
    (32, 32, 3)
    None
    (531, 800, 3)
    (32, 32, 3)
    None
    (797, 634, 3)
    (32, 32, 3)
    None
    (358, 595, 3)
    (32, 32, 3)
    None
    (1074, 894, 3)
    (32, 32, 3)
    None
    (445, 445, 3)
    (32, 32, 3)
    None
    (664, 1000, 3)
    (32, 32, 3)
    None
    (667, 1000, 3)
    (32, 32, 3)
    None
    (323, 450, 3)
    (32, 32, 3)
    None
    (696, 1054, 3)
    (32, 32, 3)
    None
    (1275, 951, 3)
    (32, 32, 3)
    None
    (434, 526, 3)
    (32, 32, 3)
    None
    (768, 1024, 3)
    (32, 32, 3)
    None
    (431, 650, 3)
    (32, 32, 3)
    None
    (477, 450, 3)
    (32, 32, 3)
    None
    (806, 1134, 3)
    (32, 32, 3)
    None
    (300, 450, 3)
    (32, 32, 3)
    None
    (428, 650, 3)
    (32, 32, 3)
    None
    (295, 450, 3)
    (32, 32, 3)
    None
    (358, 595, 3)
    (32, 32, 3)
    None
    (400, 600, 3)
    (32, 32, 3)
    None
    (800, 532, 3)
    (32, 32, 3)
    None
    (367, 550, 3)
    (32, 32, 3)
    None
    (434, 650, 3)
    (32, 32, 3)
    None
    (664, 1024, 3)
    (32, 32, 3)
    None
    (1608, 2412, 3)
    (32, 32, 3)
    None
    (853, 1280, 3)
    (32, 32, 3)
    None
    (800, 1200, 3)
    (32, 32, 3)
    None
    (512, 640, 3)
    (32, 32, 3)
    None
    (482, 366, 3)
    (32, 32, 3)
    None
    (400, 600, 3)
    (32, 32, 3)
    None
    (402, 281, 3)
    (32, 32, 3)
    None
    (1456, 2184, 3)
    (32, 32, 3)
    None
    (767, 1023, 3)
    (32, 32, 3)
    None
    (423, 650, 3)
    (32, 32, 3)
    None
    (332, 400, 3)
    (32, 32, 3)
    None
    (700, 458, 3)
    (32, 32, 3)
    None
    (524, 330, 3)
    (32, 32, 3)
    None
    (981, 1323, 3)
    (32, 32, 3)
    None
    (800, 531, 3)
    (32, 32, 3)
    None
    (862, 646, 3)
    (32, 32, 3)
    None
    (370, 470, 3)
    (32, 32, 3)
    None
    (300, 450, 3)
    (32, 32, 3)
    None
    (291, 366, 3)
    (32, 32, 3)
    None
    (607, 800, 3)
    (32, 32, 3)
    None
    (443, 291, 3)
    (32, 32, 3)
    None
    (750, 673, 3)
    (32, 32, 3)
    None
    (550, 367, 3)
    (32, 32, 3)
    None
    (456, 666, 3)
    (32, 32, 3)
    None
    (432, 650, 3)
    (32, 32, 3)
    None
    (924, 764, 3)
    (32, 32, 3)
    None
    (367, 550, 3)
    (32, 32, 3)
    None
    (640, 436, 3)
    (32, 32, 3)
    None
    (769, 1024, 3)
    (32, 32, 3)
    None
    (450, 300, 3)
    (32, 32, 3)
    None
    (533, 800, 3)
    (32, 32, 3)
    None
    (600, 900, 3)
    (32, 32, 3)
    None
    (550, 367, 3)
    (32, 32, 3)
    None
    (374, 704, 3)
    (32, 32, 3)
    None
    (819, 1024, 3)
    (32, 32, 3)
    None
    (1894, 2841, 3)
    (32, 32, 3)
    None
    (466, 500, 3)
    (32, 32, 3)
    None
    (405, 700, 3)
    (32, 32, 3)
    None
    (3456, 4608, 3)
    (32, 32, 3)
    None
    (848, 1200, 3)
    (32, 32, 3)
    None
    (577, 639, 3)
    (32, 32, 3)
    None
    (347, 649, 3)
    (32, 32, 3)
    None
    (810, 1080, 3)
    (32, 32, 3)
    None
    (554, 444, 3)
    (32, 32, 3)
    None
    (1301, 878, 3)
    (32, 32, 3)
    None
    (466, 425, 3)
    (32, 32, 3)
    None
    (1120, 1200, 3)
    (32, 32, 3)
    None
    (600, 800, 3)
    (32, 32, 3)
    None
    (626, 1024, 3)
    (32, 32, 3)
    None
    (480, 720, 3)
    (32, 32, 3)
    None
    (375, 500, 3)
    (32, 32, 3)
    None
    (532, 948, 3)
    (32, 32, 3)
    None
    (848, 1200, 3)
    (32, 32, 3)
    None
    (336, 480, 3)
    (32, 32, 3)
    None
    (2587, 3860, 3)
    (32, 32, 3)
    None
    (1068, 1600, 3)
    (32, 32, 3)
    None
    (848, 1200, 3)
    (32, 32, 3)
    None
    (533, 800, 3)
    (32, 32, 3)
    None
    (2304, 3456, 3)
    (32, 32, 3)
    None
    (450, 600, 3)
    (32, 32, 3)
    None
    (466, 700, 3)
    (32, 32, 3)
    None
    (331, 500, 3)
    (32, 32, 3)
    None
    (375, 500, 3)
    (32, 32, 3)
    None
    (386, 700, 3)
    (32, 32, 3)
    None
    (474, 360, 3)
    (32, 32, 3)
    None
    (800, 800, 3)
    (32, 32, 3)
    None
    (690, 1024, 3)
    (32, 32, 3)
    None
    (3456, 4608, 3)
    (32, 32, 3)
    None
    (1307, 866, 3)
    (32, 32, 3)
    None
    (512, 381, 3)
    (32, 32, 3)
    None
    (517, 336, 3)
    (32, 32, 3)
    None
    (371, 495, 3)
    (32, 32, 3)
    None
    (1024, 678, 3)
    (32, 32, 3)
    None
    (946, 1301, 3)
    (32, 32, 3)
    None
    (2304, 3456, 3)
    (32, 32, 3)
    None
    (900, 1200, 3)
    (32, 32, 3)
    None
    (946, 1301, 3)
    (32, 32, 3)
    None
    (683, 1024, 3)
    (32, 32, 3)
    None
    (768, 1024, 3)
    (32, 32, 3)
    None
    (349, 353, 3)
    (32, 32, 3)
    None
    (313, 500, 3)
    (32, 32, 3)
    None
    (1120, 1200, 3)
    (32, 32, 3)
    None
    (442, 500, 3)
    (32, 32, 3)
    None
    (440, 660, 3)
    (32, 32, 3)
    None
    (880, 941, 3)
    (32, 32, 3)
    None
    (615, 1024, 3)
    (32, 32, 3)
    None
    (303, 500, 3)
    (32, 32, 3)
    None
    (615, 1024, 3)
    (32, 32, 3)
    None
    (440, 467, 3)
    (32, 32, 3)
    None
    (621, 1024, 3)
    (32, 32, 3)
    None
    (2304, 3456, 3)
    (32, 32, 3)
    None
    (422, 640, 3)
    (32, 32, 3)
    None
    (480, 720, 3)
    (32, 32, 3)
    None
    (700, 601, 3)
    (32, 32, 3)
    None
    (320, 480, 3)
    (32, 32, 3)
    None
    (387, 700, 3)
    (32, 32, 3)
    None
    (480, 480, 3)
    (32, 32, 3)
    None
    (1197, 1024, 3)
    (32, 32, 3)
    None
    (612, 612, 3)
    (32, 32, 3)
    None
    (349, 353, 3)
    (32, 32, 3)
    None
    (370, 700, 3)
    (32, 32, 3)
    None
    (800, 1200, 3)
    (32, 32, 3)
    None
    (425, 295, 3)
    (32, 32, 3)
    None
    (519, 349, 3)
    (32, 32, 3)
    None
    (380, 700, 3)
    (32, 32, 3)
    None
    (398, 700, 3)
    (32, 32, 3)
    None
    (418, 623, 3)
    (32, 32, 3)
    None
    (2074, 1872, 3)
    (32, 32, 3)
    None
    (414, 537, 3)
    (32, 32, 3)
    None
    (872, 1300, 3)
    (32, 32, 3)
    None
    (731, 1300, 3)
    (32, 32, 3)
    None
    (517, 620, 3)
    (32, 32, 3)
    None
    (500, 500, 3)
    (32, 32, 3)
    None
    (1500, 2000, 3)
    (32, 32, 3)
    None
    (449, 600, 3)
    (32, 32, 3)
    None
    (592, 800, 3)
    (32, 32, 3)
    None
    (1000, 1500, 3)
    (32, 32, 3)
    None
    (344, 460, 3)
    (32, 32, 3)
    None
    (898, 1298, 3)
    (32, 32, 3)
    None
    (394, 700, 3)
    (32, 32, 3)
    None
    (1024, 1024, 3)
    (32, 32, 3)
    None
    (2304, 3456, 3)
    (32, 32, 3)
    None
    (513, 403, 3)
    (32, 32, 3)
    None
    (640, 425, 3)
    (32, 32, 3)
    None
    (660, 440, 3)
    (32, 32, 3)
    None
    (2736, 3648, 3)
    (32, 32, 3)
    None
    (640, 960, 3)
    (32, 32, 3)
    None
    (1067, 1600, 3)
    (32, 32, 3)
    None
    (640, 425, 3)
    (32, 32, 3)
    None
    (847, 1024, 3)
    (32, 32, 3)
    None
    (1288, 869, 3)
    (32, 32, 3)
    None
    (2304, 3456, 3)
    (32, 32, 3)
    None
    (426, 640, 3)
    (32, 32, 3)
    None
    (574, 1024, 3)
    (32, 32, 3)
    None
    (379, 700, 3)
    (32, 32, 3)
    None
    (393, 700, 3)
    (32, 32, 3)
    None
    (2022, 2391, 3)
    (32, 32, 3)
    None
    (619, 700, 3)
    (32, 32, 3)
    None
    (1024, 748, 3)
    (32, 32, 3)
    None
    (658, 700, 3)
    (32, 32, 3)
    None
    (1120, 1200, 3)
    (32, 32, 3)
    None
    (462, 640, 3)
    (32, 32, 3)
    None
    (600, 783, 3)
    (32, 32, 3)
    None
    (366, 550, 3)
    (32, 32, 3)
    None
    (520, 780, 3)
    (32, 32, 3)
    None
    (497, 346, 3)
    (32, 32, 3)
    None
    (500, 800, 3)
    (32, 32, 3)
    None
    (1004, 1300, 3)
    (32, 32, 3)
    None
    (519, 346, 3)
    (32, 32, 3)
    None
    (772, 1024, 3)
    (32, 32, 3)
    None
    (1298, 866, 3)
    (32, 32, 3)
    None
    (700, 560, 3)
    (32, 32, 3)
    None
    (431, 648, 3)
    (32, 32, 3)
    None
    (3648, 5472, 3)
    (32, 32, 3)
    None
    (422, 320, 3)
    (32, 32, 3)
    None
    (389, 592, 3)
    (32, 32, 3)
    None
    (500, 331, 3)
    (32, 32, 3)
    None
    (683, 1024, 3)
    (32, 32, 3)
    None
    (865, 1300, 3)
    (32, 32, 3)
    None
    (599, 398, 3)
    (32, 32, 3)
    None
    (1298, 866, 3)
    (32, 32, 3)
    None
    (601, 736, 3)
    (32, 32, 3)
    None
    (425, 597, 3)
    (32, 32, 3)
    None
    (416, 550, 3)
    (32, 32, 3)
    None
    (650, 433, 3)
    (32, 32, 3)
    None
    (375, 700, 3)
    (32, 32, 3)
    None
    (700, 500, 3)
    (32, 32, 3)
    None
    (750, 501, 3)
    (32, 32, 3)
    None
    (342, 650, 3)
    (32, 32, 3)
    None
    (596, 800, 3)
    (32, 32, 3)
    None
    (867, 1300, 3)
    (32, 32, 3)
    None
    (826, 1200, 3)
    (32, 32, 3)
    None
    (300, 450, 3)
    (32, 32, 3)
    None
    (1005, 1300, 3)
    (32, 32, 3)
    None
    (298, 450, 3)
    (32, 32, 3)
    None
    (439, 439, 3)
    (32, 32, 3)
    None
    (1174, 1200, 3)
    (32, 32, 3)
    None
    (535, 415, 3)
    (32, 32, 3)
    None
    (427, 640, 3)
    (32, 32, 3)
    None
    (369, 660, 3)
    (32, 32, 3)
    None
    (371, 450, 3)
    (32, 32, 3)
    None
    (866, 1300, 3)
    (32, 32, 3)
    None
    (428, 600, 3)
    (32, 32, 3)
    None
    (480, 640, 3)
    (32, 32, 3)
    None
    (585, 750, 3)
    (32, 32, 3)
    None
    (449, 300, 3)
    (32, 32, 3)
    None
    (866, 1300, 3)
    (32, 32, 3)
    None
    (419, 640, 3)
    (32, 32, 3)
    None
    (565, 640, 3)
    (32, 32, 3)
    None
    (450, 599, 3)
    (32, 32, 3)
    None
    (427, 640, 3)
    (32, 32, 3)
    None
    (864, 1300, 3)
    (32, 32, 3)
    None
    (599, 423, 3)
    (32, 32, 3)
    None
    (557, 743, 3)
    (32, 32, 3)
    None
    (867, 1300, 3)
    (32, 32, 3)
    None
    (382, 480, 3)
    (32, 32, 3)
    None
    (869, 1024, 3)
    (32, 32, 3)
    None
    (348, 650, 3)
    (32, 32, 3)
    None
    (920, 736, 3)
    (32, 32, 3)
    None
    (366, 467, 3)
    (32, 32, 3)
    None
    (890, 1300, 3)
    (32, 32, 3)
    None
    (599, 450, 3)
    (32, 32, 3)
    None
    (866, 1300, 3)
    (32, 32, 3)
    None
    (616, 1024, 3)
    (32, 32, 3)
    None
    (3473, 3473, 3)
    (32, 32, 3)
    None
    (344, 650, 3)
    (32, 32, 3)
    None
    (1174, 1200, 3)
    (32, 32, 3)
    None
    (298, 450, 3)
    (32, 32, 3)
    None
    (585, 800, 3)
    (32, 32, 3)
    None
    (400, 544, 3)
    (32, 32, 3)
    None
    (600, 800, 3)
    (32, 32, 3)
    None
    (600, 600, 3)
    (32, 32, 3)
    None
    (500, 800, 3)
    (32, 32, 3)
    None
    (3264, 4928, 3)
    (32, 32, 3)
    None
    (500, 800, 3)
    (32, 32, 3)
    None
    (500, 800, 3)
    (32, 32, 3)
    None
    (297, 450, 3)
    (32, 32, 3)
    None
    (683, 1024, 3)
    (32, 32, 3)
    None
    (866, 1300, 3)
    (32, 32, 3)
    None
    (572, 857, 3)
    (32, 32, 3)
    None
    (450, 600, 3)
    (32, 32, 3)
    None
    (2257, 3010, 3)
    (32, 32, 3)
    None
    (1365, 2048, 3)
    (32, 32, 3)
    None
    (600, 783, 3)
    (32, 32, 3)
    None
    (1300, 1260, 3)
    (32, 32, 3)
    None
    (422, 320, 3)
    (32, 32, 3)
    None
    (533, 800, 3)
    (32, 32, 3)
    None
    (1080, 1920, 3)
    (32, 32, 3)
    None
    (370, 470, 3)
    (32, 32, 3)
    None
    (412, 550, 3)
    (32, 32, 3)
    None
    (346, 650, 3)
    (32, 32, 3)
    None
    (366, 550, 3)
    (32, 32, 3)
    None
    (599, 798, 3)
    (32, 32, 3)
    None
    (378, 400, 3)
    (32, 32, 3)
    None
    (650, 433, 3)
    (32, 32, 3)
    None
    (480, 852, 3)
    (32, 32, 3)
    None
    (333, 500, 3)
    (32, 32, 3)
    None
    (600, 800, 3)
    (32, 32, 3)
    None
    (930, 1300, 3)
    (32, 32, 3)
    None
    (500, 800, 3)
    (32, 32, 3)
    None
    (500, 333, 3)
    (32, 32, 3)
    None
    (900, 596, 3)
    (32, 32, 3)
    None
    (1024, 1506, 3)
    (32, 32, 3)
    None
    (535, 396, 3)
    (32, 32, 3)
    None
    (900, 600, 3)
    (32, 32, 3)
    None
    (500, 750, 3)
    (32, 32, 3)
    None
    (1299, 866, 3)
    (32, 32, 3)
    None
    (533, 800, 3)
    (32, 32, 3)
    None
    (348, 650, 3)
    (32, 32, 3)
    None
    (400, 400, 3)
    (32, 32, 3)
    None
    (600, 336, 3)
    (32, 32, 3)
    None
    (384, 550, 3)
    (32, 32, 3)
    None
    (557, 743, 3)
    (32, 32, 3)
    None
    (382, 480, 3)
    (32, 32, 3)
    None
    (358, 550, 3)
    (32, 32, 3)
    None
    (854, 1300, 3)
    (32, 32, 3)
    None
    (998, 1300, 3)
    (32, 32, 3)
    None
    (768, 768, 3)
    (32, 32, 3)
    None
    (2592, 3456, 3)
    (32, 32, 3)
    None
    (666, 1000, 3)
    (32, 32, 3)
    None
    (4873, 3320, 3)
    (32, 32, 3)
    None
    (860, 1300, 3)
    (32, 32, 3)
    None
    (1296, 866, 3)
    (32, 32, 3)
    None
    (792, 1200, 3)
    (32, 32, 3)
    None
    (1294, 866, 3)
    (32, 32, 3)
    None
    (866, 1300, 3)
    (32, 32, 3)
    None
    (526, 850, 3)
    (32, 32, 3)
    None
    (434, 346, 3)
    (32, 32, 3)
    None
    (898, 1300, 3)
    (32, 32, 3)
    None
    (731, 1024, 3)
    (32, 32, 3)
    None
    (533, 800, 3)
    (32, 32, 3)
    None
    (976, 1300, 3)
    (32, 32, 3)
    None
    (850, 1300, 3)
    (32, 32, 3)
    None
    (666, 1000, 3)
    (32, 32, 3)
    None
    (300, 450, 3)
    (32, 32, 3)
    None
    (1300, 866, 3)
    (32, 32, 3)
    None
    (4796, 2728, 3)
    (32, 32, 3)
    None
    (974, 1300, 3)
    (32, 32, 3)
    None
    (298, 450, 3)
    (32, 32, 3)
    None
    (382, 300, 3)
    (32, 32, 3)
    None
    (436, 338, 3)
    (32, 32, 3)
    None
    (344, 650, 3)
    (32, 32, 3)
    None
    (866, 1300, 3)
    (32, 32, 3)
    None
    (513, 361, 3)
    (32, 32, 3)
    None
    (367, 550, 3)
    (32, 32, 3)
    None
    (600, 900, 3)
    (32, 32, 3)
    None
    (564, 564, 3)
    (32, 32, 3)
    None
    (858, 1300, 3)
    (32, 32, 3)
    None
    (1329, 2000, 3)
    (32, 32, 3)
    None
    (863, 900, 3)
    (32, 32, 3)
    None
    (1292, 1054, 3)
    (32, 32, 3)
    None
    (374, 704, 3)
    (32, 32, 3)
    None
    (3409, 2864, 3)
    (32, 32, 3)
    None
    (1288, 890, 3)
    (32, 32, 3)
    None
    (854, 1300, 3)
    (32, 32, 3)
    None
    (340, 650, 3)
    (32, 32, 3)
    None
    (862, 1300, 3)
    (32, 32, 3)
    None
    (600, 600, 3)
    (32, 32, 3)
    None
    (652, 1023, 3)
    (32, 32, 3)
    None
    (2432, 3648, 3)
    (32, 32, 3)
    None
    (1952, 3104, 3)
    (32, 32, 3)
    None
    (866, 1300, 3)
    (32, 32, 3)
    None
    (600, 900, 3)
    (32, 32, 3)
    None
    (720, 1024, 3)
    (32, 32, 3)
    None
    (336, 450, 3)
    (32, 32, 3)
    None
    (475, 420, 3)
    (32, 32, 3)
    None
    (865, 1300, 3)
    (32, 32, 3)
    None
    (374, 704, 3)
    (32, 32, 3)
    None
    (684, 1024, 3)
    (32, 32, 3)
    None
    (540, 849, 3)
    (32, 32, 3)
    None
    (501, 862, 3)
    (32, 32, 3)
    None
    (2592, 3456, 3)
    (32, 32, 3)
    None
    (3409, 2864, 3)
    (32, 32, 3)
    None
    (2976, 3968, 3)
    (32, 32, 3)
    None
    (1290, 1060, 3)
    (32, 32, 3)
    None
    (1012, 1300, 3)
    (32, 32, 3)
    None
    (682, 1024, 3)
    (32, 32, 3)
    None
    (725, 578, 3)
    (32, 32, 3)
    None
    (700, 700, 3)
    (32, 32, 3)
    None
    (298, 450, 3)
    (32, 32, 3)
    None
    (377, 568, 3)
    (32, 32, 3)
    None
    (332, 500, 3)
    (32, 32, 3)
    None
    (1772, 1152, 3)
    (32, 32, 3)
    None
    (432, 522, 3)
    (32, 32, 3)
    None
    (522, 850, 3)
    (32, 32, 3)
    None
    (486, 750, 3)
    (32, 32, 3)
    None
    (600, 390, 3)
    (32, 32, 3)
    None
    (682, 1023, 3)
    (32, 32, 3)
    None
    (1496, 1050, 3)
    (32, 32, 3)
    None
    (412, 345, 3)
    (32, 32, 3)
    None
    (1294, 848, 3)
    (32, 32, 3)
    None
    (866, 1300, 3)
    (32, 32, 3)
    None
    (1004, 1300, 3)
    (32, 32, 3)
    None
    (416, 347, 3)
    (32, 32, 3)
    None
    (683, 1024, 3)
    (32, 32, 3)
    None
    (1928, 2808, 3)
    (32, 32, 3)
    None
    (3456, 2592, 3)
    (32, 32, 3)
    None
    (907, 1300, 3)
    (32, 32, 3)
    None
    (900, 1200, 3)
    (32, 32, 3)
    None
    (361, 570, 3)
    (32, 32, 3)
    None
    (846, 1400, 3)
    (32, 32, 3)
    None
    (337, 450, 3)
    (32, 32, 3)
    None
    (682, 1023, 3)
    (32, 32, 3)
    None
    (750, 498, 3)
    (32, 32, 3)
    None
    (1800, 1200, 3)
    (32, 32, 3)
    None
    (518, 346, 3)
    (32, 32, 3)
    None
    (768, 1024, 3)
    (32, 32, 3)
    None
    (866, 1300, 3)
    (32, 32, 3)
    None
    (866, 1300, 3)
    (32, 32, 3)
    None
    (1302, 866, 3)
    (32, 32, 3)
    None
    (565, 400, 3)
    (32, 32, 3)
    None
    (300, 450, 3)
    (32, 32, 3)
    None
    (1300, 975, 3)
    (32, 32, 3)
    None
    (1800, 1200, 3)
    (32, 32, 3)
    None
    (300, 450, 3)
    (32, 32, 3)
    None
    (660, 800, 3)
    (32, 32, 3)
    None
    (866, 1300, 3)
    (32, 32, 3)
    None
    (450, 600, 3)
    (32, 32, 3)
    None
    (433, 522, 3)
    (32, 32, 3)
    None
    (661, 664, 3)
    (32, 32, 3)
    None
    (866, 1300, 3)
    (32, 32, 3)
    None
    (5472, 3648, 3)
    (32, 32, 3)
    None
    (527, 799, 3)
    (32, 32, 3)
    None
    (640, 471, 3)
    (32, 32, 3)
    None
    (600, 400, 3)
    (32, 32, 3)
    None
    (683, 1024, 3)
    (32, 32, 3)
    None
    (500, 480, 3)
    (32, 32, 3)
    None
    (735, 800, 3)
    (32, 32, 3)
    None
    (640, 1157, 3)
    (32, 32, 3)
    None
    (640, 526, 3)
    (32, 32, 3)
    None
    (450, 500, 3)
    (32, 32, 3)
    None
    (500, 387, 3)
    (32, 32, 3)
    None
    (800, 542, 3)
    (32, 32, 3)
    None
    (1824, 1959, 3)
    (32, 32, 3)
    None
    (339, 453, 3)
    (32, 32, 3)
    None
    (668, 598, 3)
    (32, 32, 3)
    None
    (2745, 4071, 3)
    (32, 32, 3)
    None
    (409, 580, 3)
    (32, 32, 3)
    None
    (376, 500, 3)
    (32, 32, 3)
    None
    (558, 368, 3)
    (32, 32, 3)
    None
    (500, 383, 3)
    (32, 32, 3)
    None
    (5472, 3648, 3)
    (32, 32, 3)
    None
    (684, 1024, 3)
    (32, 32, 3)
    None
    (3000, 2400, 3)
    (32, 32, 3)
    None
    (428, 640, 3)
    (32, 32, 3)
    None
    (843, 1292, 3)
    (32, 32, 3)
    None
    (574, 900, 3)
    (32, 32, 3)
    None
    (444, 800, 3)
    (32, 32, 3)
    None
    (332, 449, 3)
    (32, 32, 3)
    None
    (331, 452, 3)
    (32, 32, 3)
    None
    (409, 580, 3)
    (32, 32, 3)
    None
    (429, 647, 3)
    (32, 32, 3)
    None
    (458, 660, 3)
    (32, 32, 3)
    None
    (500, 412, 3)
    (32, 32, 3)
    None
    (433, 650, 3)
    (32, 32, 3)
    None
    (412, 550, 3)
    (32, 32, 3)
    None
    (337, 443, 3)
    (32, 32, 3)
    None
    (433, 650, 3)
    (32, 32, 3)
    None
    (533, 800, 3)
    (32, 32, 3)
    None
    (770, 1024, 3)
    (32, 32, 3)
    None
    (363, 364, 3)
    (32, 32, 3)
    None
    (800, 542, 3)
    (32, 32, 3)
    None
    (981, 1024, 3)
    (32, 32, 3)
    None
    (4320, 3240, 3)
    (32, 32, 3)
    None
    (720, 600, 3)
    (32, 32, 3)
    None
    (288, 512, 3)
    (32, 32, 3)
    None
    (422, 639, 3)
    (32, 32, 3)
    None
    (502, 618, 3)
    (32, 32, 3)
    None
    (520, 347, 3)
    (32, 32, 3)
    None
    (3147, 1383, 3)
    (32, 32, 3)
    None
    (858, 1292, 3)
    (32, 32, 3)
    None
    (640, 473, 3)
    (32, 32, 3)
    None
    (766, 1024, 3)
    (32, 32, 3)
    None
    (449, 312, 3)
    (32, 32, 3)
    None
    (485, 800, 3)
    (32, 32, 3)
    None
    (3240, 4320, 3)
    (32, 32, 3)
    None
    (828, 847, 3)
    (32, 32, 3)
    None
    (533, 800, 3)
    (32, 32, 3)
    None
    (1953, 2984, 3)
    (32, 32, 3)
    None
    (335, 447, 3)
    (32, 32, 3)
    None
    (1445, 1605, 3)
    (32, 32, 3)
    None
    (693, 854, 3)
    (32, 32, 3)
    None
    (1619, 2150, 3)
    (32, 32, 3)
    None
    (4320, 3240, 3)
    (32, 32, 3)
    None
    (706, 846, 3)
    (32, 32, 3)
    None
    (790, 907, 3)
    (32, 32, 3)
    None
    (406, 625, 3)
    (32, 32, 3)
    None
    (480, 600, 3)
    (32, 32, 3)
    None
    (693, 854, 3)
    (32, 32, 3)
    None
    (599, 383, 3)
    (32, 32, 3)
    None
    (2235, 2433, 3)
    (32, 32, 3)
    None
    (2400, 3600, 3)
    (32, 32, 3)
    None
    (866, 1162, 3)
    (32, 32, 3)
    None
    (384, 522, 3)
    (32, 32, 3)
    None
    (334, 451, 3)
    (32, 32, 3)
    None
    (406, 346, 3)
    (32, 32, 3)
    None
    (1193, 858, 3)
    (32, 32, 3)
    None
    (560, 840, 3)
    (32, 32, 3)
    None
    (316, 474, 3)
    (32, 32, 3)
    None
    (1024, 835, 3)
    (32, 32, 3)
    None
    (336, 445, 3)
    (32, 32, 3)
    None
    (600, 400, 3)
    (32, 32, 3)
    None
    (770, 500, 3)
    (32, 32, 3)
    None
    (2400, 3600, 3)
    (32, 32, 3)
    None
    (414, 537, 3)
    (32, 32, 3)
    None
    (683, 1024, 3)
    (32, 32, 3)
    None
    (2592, 4608, 3)
    (32, 32, 3)
    None
    (2889, 1875, 3)
    (32, 32, 3)
    None
    (296, 446, 3)
    (32, 32, 3)
    None
    (700, 1050, 3)
    (32, 32, 3)
    None
    (640, 463, 3)
    (32, 32, 3)
    None
    (541, 479, 3)
    (32, 32, 3)
    None
    (395, 600, 3)
    (32, 32, 3)
    None
    (453, 347, 3)
    (32, 32, 3)
    None
    (480, 480, 3)
    (32, 32, 3)
    None
    (553, 1024, 3)
    (32, 32, 3)
    None
    (552, 736, 3)
    (32, 32, 3)
    None
    (413, 628, 3)
    (32, 32, 3)
    None
    (367, 550, 3)
    (32, 32, 3)
    None
    (1024, 683, 3)
    (32, 32, 3)
    None
    (250, 444, 3)
    (32, 32, 3)
    None
    (520, 347, 3)
    (32, 32, 3)
    None
    (1159, 1920, 3)
    (32, 32, 3)
    None
    (444, 800, 3)
    (32, 32, 3)
    None
    (600, 800, 3)
    (32, 32, 3)
    None
    (750, 1000, 3)
    (32, 32, 3)
    None
    (290, 505, 3)
    (32, 32, 3)
    None



```python

```




    'n9153.jpg'




```python

```




    array([[[156, 180, 172],
            [155, 179, 171],
            [154, 178, 170],
            ...,
            [141, 161, 166],
            [147, 166, 169],
            [158, 176, 177]],
    
           [[154, 177, 172],
            [153, 176, 171],
            [151, 174, 169],
            ...,
            [140, 160, 165],
            [147, 166, 169],
            [155, 173, 174]],
    
           [[150, 173, 169],
            [150, 173, 169],
            [148, 171, 166],
            ...,
            [138, 158, 163],
            [146, 165, 168],
            [151, 169, 170]],
    
           ...,
    
           [[211, 219, 208],
            [207, 216, 206],
            [203, 211, 204],
            ...,
            [217, 233, 222],
            [224, 232, 221],
            [231, 233, 221]],
    
           [[217, 225, 214],
            [212, 220, 209],
            [204, 213, 203],
            ...,
            [230, 240, 228],
            [233, 235, 223],
            [236, 233, 219]],
    
           [[228, 237, 224],
            [220, 228, 217],
            [208, 217, 207],
            ...,
            [239, 243, 231],
            [238, 237, 223],
            [242, 233, 219]]], dtype=uint8)




```python
main(train_data, train_label)
```

    Loaded from VGG
    Setting up summary op...
    Setting Up Saver...
    epoch : 0 step : 0 train_loss : 38.82235 train_accuracy : 0.03125
    epoch : 0 step : 10 train_loss : 2.2874923 train_accuracy : 0.09375
    epoch : 0 step : 20 train_loss : 2.0633779 train_accuracy : 0.25
    epoch : 0 step : 30 train_loss : 2.0674796 train_accuracy : 0.3125
    epoch : 1 step : 0 train_loss : 1.7375543 train_accuracy : 0.53125
    epoch : 1 step : 10 train_loss : 2.2400854 train_accuracy : 0.15625
    epoch : 1 step : 20 train_loss : 1.8993624 train_accuracy : 0.3125
    epoch : 1 step : 30 train_loss : 1.624648 train_accuracy : 0.34375
    epoch : 2 step : 0 train_loss : 1.5888789 train_accuracy : 0.4375
    epoch : 2 step : 10 train_loss : 1.8040812 train_accuracy : 0.34375
    epoch : 2 step : 20 train_loss : 1.9816607 train_accuracy : 0.25
    epoch : 2 step : 30 train_loss : 2.2372465 train_accuracy : 0.21875
    epoch : 3 step : 0 train_loss : 1.3529875 train_accuracy : 0.5
    epoch : 3 step : 10 train_loss : 1.5211806 train_accuracy : 0.5
    epoch : 3 step : 20 train_loss : 1.7554313 train_accuracy : 0.375
    epoch : 3 step : 30 train_loss : 1.5471078 train_accuracy : 0.4375
    epoch : 4 step : 0 train_loss : 1.7856631 train_accuracy : 0.40625
    epoch : 4 step : 10 train_loss : 1.2698052 train_accuracy : 0.46875
    epoch : 4 step : 20 train_loss : 1.5464437 train_accuracy : 0.4375
    epoch : 4 step : 30 train_loss : 1.3577685 train_accuracy : 0.375
    epoch : 5 step : 0 train_loss : 1.3322265 train_accuracy : 0.65625
    epoch : 5 step : 10 train_loss : 1.71533 train_accuracy : 0.375
    epoch : 5 step : 20 train_loss : 1.1508808 train_accuracy : 0.59375
    epoch : 5 step : 30 train_loss : 1.4221945 train_accuracy : 0.5625
    epoch : 6 step : 0 train_loss : 1.0442168 train_accuracy : 0.625
    epoch : 6 step : 10 train_loss : 1.3753965 train_accuracy : 0.5
    epoch : 6 step : 20 train_loss : 1.0306852 train_accuracy : 0.625
    epoch : 6 step : 30 train_loss : 0.90503156 train_accuracy : 0.625
    epoch : 7 step : 0 train_loss : 1.1811953 train_accuracy : 0.625
    epoch : 7 step : 10 train_loss : 0.9265252 train_accuracy : 0.53125
    epoch : 7 step : 20 train_loss : 0.84975725 train_accuracy : 0.8125
    epoch : 7 step : 30 train_loss : 1.1335437 train_accuracy : 0.71875
    epoch : 8 step : 0 train_loss : 0.8871374 train_accuracy : 0.6875
    epoch : 8 step : 10 train_loss : 1.2456572 train_accuracy : 0.59375
    epoch : 8 step : 20 train_loss : 0.7631475 train_accuracy : 0.71875
    epoch : 8 step : 30 train_loss : 0.8386714 train_accuracy : 0.78125
    epoch : 9 step : 0 train_loss : 0.40604538 train_accuracy : 0.90625
    epoch : 9 step : 10 train_loss : 0.85728574 train_accuracy : 0.75
    epoch : 9 step : 20 train_loss : 0.9295616 train_accuracy : 0.75
    epoch : 9 step : 30 train_loss : 0.722451 train_accuracy : 0.75
    epoch : 10 step : 0 train_loss : 0.97234654 train_accuracy : 0.65625
    epoch : 10 step : 10 train_loss : 0.5812726 train_accuracy : 0.875
    epoch : 10 step : 20 train_loss : 0.73473275 train_accuracy : 0.75
    epoch : 10 step : 30 train_loss : 0.5186591 train_accuracy : 0.8125
    epoch : 11 step : 0 train_loss : 0.5259063 train_accuracy : 0.78125
    epoch : 11 step : 10 train_loss : 0.60489184 train_accuracy : 0.78125
    epoch : 11 step : 20 train_loss : 0.293837 train_accuracy : 0.9375
    epoch : 11 step : 30 train_loss : 0.63365906 train_accuracy : 0.78125
    epoch : 12 step : 0 train_loss : 0.39980012 train_accuracy : 0.875
    epoch : 12 step : 10 train_loss : 0.45136458 train_accuracy : 0.875
    epoch : 12 step : 20 train_loss : 0.4545532 train_accuracy : 0.84375
    epoch : 12 step : 30 train_loss : 0.41464156 train_accuracy : 0.875
    epoch : 13 step : 0 train_loss : 0.41553363 train_accuracy : 0.875
    epoch : 13 step : 10 train_loss : 0.10099784 train_accuracy : 1.0
    epoch : 13 step : 20 train_loss : 1.1447842 train_accuracy : 0.625
    epoch : 13 step : 30 train_loss : 0.4404956 train_accuracy : 0.8125
    epoch : 14 step : 0 train_loss : 0.3145449 train_accuracy : 0.90625
    epoch : 14 step : 10 train_loss : 0.11697939 train_accuracy : 1.0
    epoch : 14 step : 20 train_loss : 0.097085565 train_accuracy : 1.0
    epoch : 14 step : 30 train_loss : 0.45987317 train_accuracy : 0.84375
    epoch : 15 step : 0 train_loss : 0.2889766 train_accuracy : 0.875
    epoch : 15 step : 10 train_loss : 0.15157469 train_accuracy : 0.96875
    epoch : 15 step : 20 train_loss : 0.19333099 train_accuracy : 0.96875
    epoch : 15 step : 30 train_loss : 0.089888684 train_accuracy : 1.0
    epoch : 16 step : 0 train_loss : 0.09311714 train_accuracy : 0.96875
    epoch : 16 step : 10 train_loss : 0.1079946 train_accuracy : 0.96875
    epoch : 16 step : 20 train_loss : 0.10505624 train_accuracy : 0.96875
    epoch : 16 step : 30 train_loss : 0.23592772 train_accuracy : 0.9375
    epoch : 17 step : 0 train_loss : 0.31307754 train_accuracy : 0.90625
    epoch : 17 step : 10 train_loss : 0.41944444 train_accuracy : 0.78125
    epoch : 17 step : 20 train_loss : 0.119749814 train_accuracy : 0.96875
    epoch : 17 step : 30 train_loss : 0.06212578 train_accuracy : 0.96875
    epoch : 18 step : 0 train_loss : 0.031461015 train_accuracy : 1.0
    epoch : 18 step : 10 train_loss : 0.05148617 train_accuracy : 1.0
    epoch : 18 step : 20 train_loss : 0.08023304 train_accuracy : 0.96875
    epoch : 18 step : 30 train_loss : 0.05528934 train_accuracy : 0.96875
    epoch : 19 step : 0 train_loss : 0.04669997 train_accuracy : 1.0
    epoch : 19 step : 10 train_loss : 0.035735272 train_accuracy : 1.0
    epoch : 19 step : 20 train_loss : 0.020655507 train_accuracy : 1.0
    epoch : 19 step : 30 train_loss : 0.05165346 train_accuracy : 1.0
    epoch : 20 step : 0 train_loss : 0.02742078 train_accuracy : 1.0
    epoch : 20 step : 10 train_loss : 0.018845014 train_accuracy : 1.0
    epoch : 20 step : 20 train_loss : 0.014720585 train_accuracy : 1.0
    epoch : 20 step : 30 train_loss : 0.034627087 train_accuracy : 1.0
    epoch : 21 step : 0 train_loss : 0.0110759735 train_accuracy : 1.0
    epoch : 21 step : 10 train_loss : 0.012583188 train_accuracy : 1.0
    epoch : 21 step : 20 train_loss : 0.009513136 train_accuracy : 1.0
    epoch : 21 step : 30 train_loss : 0.016461106 train_accuracy : 1.0
    epoch : 22 step : 0 train_loss : 0.007528426 train_accuracy : 1.0
    epoch : 22 step : 10 train_loss : 0.0070482953 train_accuracy : 1.0
    epoch : 22 step : 20 train_loss : 0.007969259 train_accuracy : 1.0
    epoch : 22 step : 30 train_loss : 0.006491976 train_accuracy : 1.0
    epoch : 23 step : 0 train_loss : 0.0049894955 train_accuracy : 1.0
    epoch : 23 step : 10 train_loss : 0.009445796 train_accuracy : 1.0
    epoch : 23 step : 20 train_loss : 0.0123352185 train_accuracy : 1.0
    epoch : 23 step : 30 train_loss : 0.022943424 train_accuracy : 1.0
    epoch : 24 step : 0 train_loss : 0.01627249 train_accuracy : 1.0
    epoch : 24 step : 10 train_loss : 0.002990456 train_accuracy : 1.0
    epoch : 24 step : 20 train_loss : 0.007661726 train_accuracy : 1.0
    epoch : 24 step : 30 train_loss : 0.0060836407 train_accuracy : 1.0
    epoch : 25 step : 0 train_loss : 0.0055635935 train_accuracy : 1.0
    epoch : 25 step : 10 train_loss : 0.005115156 train_accuracy : 1.0
    epoch : 25 step : 20 train_loss : 0.0053334935 train_accuracy : 1.0
    epoch : 25 step : 30 train_loss : 0.0034122365 train_accuracy : 1.0
    epoch : 26 step : 0 train_loss : 0.004042819 train_accuracy : 1.0
    epoch : 26 step : 10 train_loss : 0.009596109 train_accuracy : 1.0
    epoch : 26 step : 20 train_loss : 0.0048308037 train_accuracy : 1.0
    epoch : 26 step : 30 train_loss : 0.0067915786 train_accuracy : 1.0
    epoch : 27 step : 0 train_loss : 0.0046366975 train_accuracy : 1.0
    epoch : 27 step : 10 train_loss : 0.0047340416 train_accuracy : 1.0
    epoch : 27 step : 20 train_loss : 0.0042392593 train_accuracy : 1.0
    epoch : 27 step : 30 train_loss : 0.0037874973 train_accuracy : 1.0
    epoch : 28 step : 0 train_loss : 0.002124194 train_accuracy : 1.0
    epoch : 28 step : 10 train_loss : 0.0050974656 train_accuracy : 1.0
    epoch : 28 step : 20 train_loss : 0.0024186913 train_accuracy : 1.0
    epoch : 28 step : 30 train_loss : 0.0062831896 train_accuracy : 1.0
    epoch : 29 step : 0 train_loss : 0.0026063058 train_accuracy : 1.0
    epoch : 29 step : 10 train_loss : 0.0042290464 train_accuracy : 1.0
    epoch : 29 step : 20 train_loss : 0.0018717526 train_accuracy : 1.0
    epoch : 29 step : 30 train_loss : 0.006727062 train_accuracy : 1.0
    epoch : 30 step : 0 train_loss : 0.0024514515 train_accuracy : 1.0
    epoch : 30 step : 10 train_loss : 0.0045481524 train_accuracy : 1.0
    epoch : 30 step : 20 train_loss : 0.0011262574 train_accuracy : 1.0
    epoch : 30 step : 30 train_loss : 0.005561225 train_accuracy : 1.0
    epoch : 31 step : 0 train_loss : 0.0022283965 train_accuracy : 1.0
    epoch : 31 step : 10 train_loss : 0.002258006 train_accuracy : 1.0
    epoch : 31 step : 20 train_loss : 0.0042035654 train_accuracy : 1.0
    epoch : 31 step : 30 train_loss : 0.0021608174 train_accuracy : 1.0
    epoch : 32 step : 0 train_loss : 0.0027458845 train_accuracy : 1.0
    epoch : 32 step : 10 train_loss : 0.0032542248 train_accuracy : 1.0
    epoch : 32 step : 20 train_loss : 0.0025711425 train_accuracy : 1.0
    epoch : 32 step : 30 train_loss : 0.0026641479 train_accuracy : 1.0
    epoch : 33 step : 0 train_loss : 0.0024669855 train_accuracy : 1.0
    epoch : 33 step : 10 train_loss : 0.0037071486 train_accuracy : 1.0
    epoch : 33 step : 20 train_loss : 0.0024550362 train_accuracy : 1.0
    epoch : 33 step : 30 train_loss : 0.0038059372 train_accuracy : 1.0
    epoch : 34 step : 0 train_loss : 0.0012781249 train_accuracy : 1.0
    epoch : 34 step : 10 train_loss : 0.0048444737 train_accuracy : 1.0
    epoch : 34 step : 20 train_loss : 0.0019913875 train_accuracy : 1.0
    epoch : 34 step : 30 train_loss : 0.0025739819 train_accuracy : 1.0
    epoch : 35 step : 0 train_loss : 0.00243926 train_accuracy : 1.0
    epoch : 35 step : 10 train_loss : 0.0025789188 train_accuracy : 1.0
    epoch : 35 step : 20 train_loss : 0.0017379228 train_accuracy : 1.0
    epoch : 35 step : 30 train_loss : 0.001105392 train_accuracy : 1.0
    epoch : 36 step : 0 train_loss : 0.0026557907 train_accuracy : 1.0
    epoch : 36 step : 10 train_loss : 0.00265283 train_accuracy : 1.0
    epoch : 36 step : 20 train_loss : 0.0028593964 train_accuracy : 1.0
    epoch : 36 step : 30 train_loss : 0.0018537068 train_accuracy : 1.0
    epoch : 37 step : 0 train_loss : 0.0027075103 train_accuracy : 1.0
    epoch : 37 step : 10 train_loss : 0.0020257526 train_accuracy : 1.0
    epoch : 37 step : 20 train_loss : 0.0018306841 train_accuracy : 1.0
    epoch : 37 step : 30 train_loss : 0.0014013557 train_accuracy : 1.0
    epoch : 38 step : 0 train_loss : 0.0016408003 train_accuracy : 1.0
    epoch : 38 step : 10 train_loss : 0.0014853757 train_accuracy : 1.0
    epoch : 38 step : 20 train_loss : 0.0015123218 train_accuracy : 1.0
    epoch : 38 step : 30 train_loss : 0.0021694994 train_accuracy : 1.0
    epoch : 39 step : 0 train_loss : 0.0014355562 train_accuracy : 1.0
    epoch : 39 step : 10 train_loss : 0.0018668654 train_accuracy : 1.0
    epoch : 39 step : 20 train_loss : 0.0014042272 train_accuracy : 1.0
    epoch : 39 step : 30 train_loss : 0.001019021 train_accuracy : 1.0
    epoch : 40 step : 0 train_loss : 0.001483769 train_accuracy : 1.0
    epoch : 40 step : 10 train_loss : 0.0007514729 train_accuracy : 1.0
    epoch : 40 step : 20 train_loss : 0.0010717218 train_accuracy : 1.0
    epoch : 40 step : 30 train_loss : 0.0028465642 train_accuracy : 1.0
    epoch : 41 step : 0 train_loss : 0.0011058997 train_accuracy : 1.0
    epoch : 41 step : 10 train_loss : 0.0017791897 train_accuracy : 1.0
    epoch : 41 step : 20 train_loss : 0.0021030963 train_accuracy : 1.0
    epoch : 41 step : 30 train_loss : 0.0019057257 train_accuracy : 1.0
    epoch : 42 step : 0 train_loss : 0.0013480056 train_accuracy : 1.0
    epoch : 42 step : 10 train_loss : 0.0014468371 train_accuracy : 1.0
    epoch : 42 step : 20 train_loss : 0.0017720071 train_accuracy : 1.0
    epoch : 42 step : 30 train_loss : 0.0015967217 train_accuracy : 1.0
    epoch : 43 step : 0 train_loss : 0.0012234927 train_accuracy : 1.0
    epoch : 43 step : 10 train_loss : 0.0012297378 train_accuracy : 1.0
    epoch : 43 step : 20 train_loss : 0.0020056718 train_accuracy : 1.0
    epoch : 43 step : 30 train_loss : 0.000651376 train_accuracy : 1.0
    epoch : 44 step : 0 train_loss : 0.0014513072 train_accuracy : 1.0
    epoch : 44 step : 10 train_loss : 0.0014020966 train_accuracy : 1.0
    epoch : 44 step : 20 train_loss : 0.000982752 train_accuracy : 1.0
    epoch : 44 step : 30 train_loss : 0.0010775742 train_accuracy : 1.0
    epoch : 45 step : 0 train_loss : 0.00065168797 train_accuracy : 1.0
    epoch : 45 step : 10 train_loss : 0.0017524391 train_accuracy : 1.0
    epoch : 45 step : 20 train_loss : 0.0011018196 train_accuracy : 1.0
    epoch : 45 step : 30 train_loss : 0.0016755671 train_accuracy : 1.0
    epoch : 46 step : 0 train_loss : 0.001272636 train_accuracy : 1.0
    epoch : 46 step : 10 train_loss : 0.00083053636 train_accuracy : 1.0
    epoch : 46 step : 20 train_loss : 0.0015635485 train_accuracy : 1.0
    epoch : 46 step : 30 train_loss : 0.0008026693 train_accuracy : 1.0
    epoch : 47 step : 0 train_loss : 0.0015117591 train_accuracy : 1.0
    epoch : 47 step : 10 train_loss : 0.0015482553 train_accuracy : 1.0
    epoch : 47 step : 20 train_loss : 0.0010718144 train_accuracy : 1.0
    epoch : 47 step : 30 train_loss : 0.0014636476 train_accuracy : 1.0
    epoch : 48 step : 0 train_loss : 0.0016296107 train_accuracy : 1.0
    epoch : 48 step : 10 train_loss : 0.00089376303 train_accuracy : 1.0
    epoch : 48 step : 20 train_loss : 0.0013037595 train_accuracy : 1.0
    epoch : 48 step : 30 train_loss : 0.0009918583 train_accuracy : 1.0
    epoch : 49 step : 0 train_loss : 0.0007959747 train_accuracy : 1.0
    epoch : 49 step : 10 train_loss : 0.0010981224 train_accuracy : 1.0
    epoch : 49 step : 20 train_loss : 0.00072687864 train_accuracy : 1.0
    epoch : 49 step : 30 train_loss : 0.0021492476 train_accuracy : 1.0
    epoch : 50 step : 0 train_loss : 0.00071069563 train_accuracy : 1.0
    epoch : 50 step : 10 train_loss : 0.0013003657 train_accuracy : 1.0
    epoch : 50 step : 20 train_loss : 0.0012905317 train_accuracy : 1.0
    epoch : 50 step : 30 train_loss : 0.0010352123 train_accuracy : 1.0
    epoch : 51 step : 0 train_loss : 0.001022313 train_accuracy : 1.0
    epoch : 51 step : 10 train_loss : 0.00074943324 train_accuracy : 1.0
    epoch : 51 step : 20 train_loss : 0.000970414 train_accuracy : 1.0
    epoch : 51 step : 30 train_loss : 0.0011123354 train_accuracy : 1.0
    epoch : 52 step : 0 train_loss : 0.00090033055 train_accuracy : 1.0
    epoch : 52 step : 10 train_loss : 0.001221513 train_accuracy : 1.0
    epoch : 52 step : 20 train_loss : 0.0017419632 train_accuracy : 1.0
    epoch : 52 step : 30 train_loss : 0.0008920692 train_accuracy : 1.0
    epoch : 53 step : 0 train_loss : 0.001182784 train_accuracy : 1.0
    epoch : 53 step : 10 train_loss : 0.000942832 train_accuracy : 1.0
    epoch : 53 step : 20 train_loss : 0.0007016316 train_accuracy : 1.0
    epoch : 53 step : 30 train_loss : 0.0012959337 train_accuracy : 1.0
    epoch : 54 step : 0 train_loss : 0.0009767683 train_accuracy : 1.0
    epoch : 54 step : 10 train_loss : 0.00084721367 train_accuracy : 1.0
    epoch : 54 step : 20 train_loss : 0.00080033194 train_accuracy : 1.0
    epoch : 54 step : 30 train_loss : 0.0008748385 train_accuracy : 1.0
    epoch : 55 step : 0 train_loss : 0.00054712256 train_accuracy : 1.0
    epoch : 55 step : 10 train_loss : 0.00069753616 train_accuracy : 1.0
    epoch : 55 step : 20 train_loss : 0.0011472232 train_accuracy : 1.0
    epoch : 55 step : 30 train_loss : 0.0013344749 train_accuracy : 1.0
    epoch : 56 step : 0 train_loss : 0.00084952044 train_accuracy : 1.0
    epoch : 56 step : 10 train_loss : 0.00037117963 train_accuracy : 1.0
    epoch : 56 step : 20 train_loss : 0.0007288269 train_accuracy : 1.0
    epoch : 56 step : 30 train_loss : 0.0015047581 train_accuracy : 1.0
    epoch : 57 step : 0 train_loss : 0.00083075475 train_accuracy : 1.0
    epoch : 57 step : 10 train_loss : 0.00061605324 train_accuracy : 1.0
    epoch : 57 step : 20 train_loss : 0.001240546 train_accuracy : 1.0
    epoch : 57 step : 30 train_loss : 0.0010310645 train_accuracy : 1.0



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-49-363e593e3c44> in <module>()
    ----> 1 main(train_data, train_label)
    

    <ipython-input-41-0e7b23875512> in main(train_data, train_label, no_of_epochs, batchsize)
         47             feed_dict = {Input : batchx , Label : batchy}
         48 
    ---> 49             _, train_loss, train_accuracy, summary_str = sess.run([optimiz, loss, accuracy, summary_op] , feed_dict )
         50             summary_writer.add_summary(summary_str, itr)
         51             itr = itr + 1


    /root/miniconda2/lib/python2.7/site-packages/tensorflow/python/client/session.pyc in run(self, fetches, feed_dict, options, run_metadata)
        903     try:
        904       result = self._run(None, fetches, feed_dict, options_ptr,
    --> 905                          run_metadata_ptr)
        906       if run_metadata:
        907         proto_data = tf_session.TF_GetBuffer(run_metadata_ptr)


    /root/miniconda2/lib/python2.7/site-packages/tensorflow/python/client/session.pyc in _run(self, handle, fetches, feed_dict, options, run_metadata)
       1138     if final_fetches or final_targets or (handle and feed_dict_tensor):
       1139       results = self._do_run(handle, final_targets, final_fetches,
    -> 1140                              feed_dict_tensor, options, run_metadata)
       1141     else:
       1142       results = []


    /root/miniconda2/lib/python2.7/site-packages/tensorflow/python/client/session.pyc in _do_run(self, handle, target_list, fetch_list, feed_dict, options, run_metadata)
       1319     if handle is None:
       1320       return self._do_call(_run_fn, feeds, fetches, targets, options,
    -> 1321                            run_metadata)
       1322     else:
       1323       return self._do_call(_prun_fn, handle, feeds, fetches)


    /root/miniconda2/lib/python2.7/site-packages/tensorflow/python/client/session.pyc in _do_call(self, fn, *args)
       1325   def _do_call(self, fn, *args):
       1326     try:
    -> 1327       return fn(*args)
       1328     except errors.OpError as e:
       1329       message = compat.as_text(e.message)


    /root/miniconda2/lib/python2.7/site-packages/tensorflow/python/client/session.pyc in _run_fn(feed_dict, fetch_list, target_list, options, run_metadata)
       1310       self._extend_graph()
       1311       return self._call_tf_sessionrun(
    -> 1312           options, feed_dict, fetch_list, target_list, run_metadata)
       1313 
       1314     def _prun_fn(handle, feed_dict, fetch_list):


    /root/miniconda2/lib/python2.7/site-packages/tensorflow/python/client/session.pyc in _call_tf_sessionrun(self, options, feed_dict, fetch_list, target_list, run_metadata)
       1418         return tf_session.TF_Run(
       1419             self._session, options, feed_dict, fetch_list, target_list,
    -> 1420             status, run_metadata)
       1421 
       1422   def _call_tf_sessionprun(self, handle, feed_dict, fetch_list):


    KeyboardInterrupt: 

