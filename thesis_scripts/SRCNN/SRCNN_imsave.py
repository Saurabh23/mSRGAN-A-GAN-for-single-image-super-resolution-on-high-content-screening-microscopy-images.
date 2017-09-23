
# coding: utf-8

# In[11]:

import matplotlib.pyplot as plt
import numpy as np
import math
from PIL import Image
import _pickle as cPickle
#import cPickle as pkl
import time
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import glob
from scipy.misc import imsave
get_ipython().magic('matplotlib inline')
print ("Packages loaded")


# # Load dataset

# In[5]:

dirpath = "C:/Users/Saurabh/Documents/resized/"
height = 240
width  = 360
resize_ratio = 4
nr_img = 0
fileList = glob.glob(dirpath + '*.jpg')
for i, file in enumerate(fileList):
    img = Image.open(file)
    array = np.array(img) 
    if array.shape[0] == height and array.shape[1] == width:
        nr_img = nr_img + 1
        rgb = array.reshape(1, height, width, 3)
        
        imglow = img.resize((int(width/resize_ratio) 
                ,int(height/resize_ratio)), Image.BICUBIC) # Reducing to 90*60 by bicubic
        #print("imglow")
        #print("THe width of imglow v1 is:",imglow.width)
        #plt.imshow(imglow)
        imglow = imglow.resize((width, height), Image.BICUBIC) # Resize to 360*240
        #print("img v2")
        #plt.imshow(imglow)
        #print("THe width of imglow v2 is:",imglow.width)
        
        rgblow = np.array(np.float32(imglow)/255.) # Converting to float 32
        rgblow = rgblow.reshape(1, height, width, 3) # Batch dimension (N * H * W * C)
        rgb = np.reshape(rgb, [1, -1])
        
        rgblow = np.reshape(rgblow, [1, -1])
        if nr_img == 1:
            data = rgb
            datalow = rgblow
        else:
            data = np.concatenate((data, rgb), axis=0)
            datalow = np.concatenate((datalow, rgblow), axis=0)
        
print ("nr_img is %d" % (nr_img))
print ("Shape of 'data' is %s" % (data.shape,))
print ("Shape of 'datalow' is %s" % (datalow.shape,))


# # Divide into two sets
# ## (xtrain, ytrain) and (xtest, ytest)

# In[6]:

randidx = np.random.permutation(nr_img)
nrtrain = int(nr_img*0.7)
nrtest  = nr_img - nrtrain
xtrain  = datalow[randidx[0:nrtrain], :]
ytrain  = data[randidx[0:nrtrain], :]
xtest   = datalow[randidx[nrtrain:nr_img], :]
ytest   = data[randidx[nrtrain:nr_img], :]
print ("Shape of 'xtrain' is %s" % (xtrain.shape,))
print ("Shape of 'ytrain' is %s" % (ytrain.shape,))
print ("Shape of 'xtest' is %s" % (xtest.shape,))
print ("Shape of 'ytest' is %s" % (ytest.shape,))


# # Plot some images

# In[7]:

# Train
randidx = np.random.randint(nrtrain)
currx = xtrain[randidx, :]
currx = np.reshape(currx, [height, width, 3])
plt.imshow(currx)
plt.title("Train input image")
plt.show()
curry = ytrain[randidx, :]
curry = np.reshape(curry, [height, width, 3])
plt.imshow(curry)
plt.title("Train output image")
plt.show() 
# Test
randidx = np.random.randint(nrtest)
currx = xtest[randidx, :]
currx = np.reshape(currx, [height, width, 3])
plt.imshow(currx)
plt.title("Test input image")
plt.show()
curry = ytest[randidx, :]
curry = np.reshape(curry, [height, width, 3])
plt.imshow(curry)
plt.title("Test output image")
plt.show()


# # Define network

# In[8]:

# Filter sizes as mentioned in the SRCNN paper
n1 = 64
n2 = 32
n3 = 3
ksize1 = 9
ksize2 = 1
ksize3 = 5


weights = {
    'ce1': tf.Variable(tf.random_normal([ksize1,ksize1,3,n1])),
    'ce2': tf.Variable(tf.random_normal([ksize2,ksize2,64,n2])),
    'ce3': tf.Variable(tf.random_normal([ksize3,ksize3,n2,n3]))
}
                      

biases = {
    'be1': tf.Variable(tf.random_normal([n1], stddev = 0.01)),   
    'be2': tf.Variable(tf.random_normal([n2], stddev = 0.01)),
    'be3': tf.Variable(tf.random_normal([n3], stddev = 0.01)),
}

def srn(_X, _W, _b, _keepprob):
    _input_r = tf.reshape(_X, shape=[-1, height, width, 3])
    
    _ce1 = tf.nn.relu(tf.add(tf.nn.conv2d(_input_r, _W['ce1']
        , strides=[1, 1, 1, 1], padding='SAME'), _b['be1']))
    _ce1 = tf.nn.dropout(_ce1, _keepprob)
    _ce2 = tf.nn.relu(tf.add(tf.nn.conv2d(_ce1, _W['ce2']
        , strides=[1, 1, 1, 1], padding='SAME'), _b['be2'])) 
    _ce2 = tf.nn.dropout(_ce2, _keepprob)
    _ce3 = tf.nn.relu(tf.add(tf.nn.conv2d(_ce2, _W['ce3']
        , strides=[1, 1, 1, 1], padding='SAME'), _b['be3'])) 
    _ce3 = tf.nn.dropout(_ce3, _keepprob)

    _out = _ce3 + _input_r                  
                      
    return {'input_r': _input_r, 'ce1': _ce1, 'ce2': _ce2, 'ce3': _ce3
        , 'layers': (_input_r, _ce1, _ce2, _ce3)
        , 'out': _out}
                      
print ("Network ready")


# # Define functions

# In[9]:

dim = height*width*3
x = tf.placeholder(tf.float32, [None, dim])
y = tf.placeholder(tf.float32, [None, dim])
keepprob = tf.placeholder(tf.float32)
pred = srn(x, weights, biases, keepprob)['out']
cost = tf.reduce_mean(tf.square(srn(x, weights, biases, keepprob)['out'] 
            - tf.reshape(y, shape=[-1, height, width, 3])))
#tf.summmary.add scaler (Tensorboard)
learning_rate = 0.001
optm = tf.train.AdamOptimizer(learning_rate, 0.9).minimize(cost)
#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
print ("Functions ready")


# # Run

# In[ ]:

sess = tf.Session()
sess.run(init)
# Fit all training data
batch_size = 1
n_epochs   = 10000
print("Strart training..")

for epoch_i in range(n_epochs):  

    for batch_i in range(nrtrain // batch_size):
        randidx = np.random.randint(nrtrain, size=batch_size)
        batch_xs = xtrain[randidx, :]
        batch_ys = ytrain[randidx, :]
        sess.run(optm, feed_dict={x: batch_xs
            , y: batch_ys, keepprob: 0.7})
        #print ("[%02d/%02d] cost: %.4f" % (epoch_i, n_epochs, sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})))
        #cost = sess.run(cost, feed_dict ={x: batch_xs, y: batch_ys})
        
    if (epoch_i % 10) == 0:
        #print (epoch_i)
        print("....") 
        #print(cost)
    if (epoch_i % 100) == 0:
        print("...........................")
        n_examples = 2
        print ("Training dataset")
        randidx = np.random.randint(nrtrain, size=n_examples)
        train_xs = xtrain[randidx, :]
        train_ys = ytrain[randidx, :]
        recon = sess.run(pred, feed_dict={x: train_xs, keepprob: 1.0})
        #print("the recon is: ",recon)
        fig, axs = plt.subplots(3, n_examples, figsize=(15, 20))
        
        for example_i in range(n_examples):
            axs[0][example_i].imshow(np.reshape(
                train_xs[example_i, :], (height, width, 3)))
            axs[1][example_i].imshow(np.reshape(
                recon[example_i, :], (height, width, 3)))
            axs[2][example_i].imshow(np.reshape(
                train_ys[example_i, :], (height, width, 3)))
        plt.show()
        print ("Test dataset")
        randidx = np.random.randint(nrtest, size=n_examples)
        test_xs = xtest[randidx, :]
        test_ys = ytest[randidx, :]
        recon = sess.run(pred, feed_dict={x: test_xs, keepprob: 1.0})
        fig, axs = plt.subplots(3, n_examples, figsize=(15, 20))
        for example_i in range(n_examples):
            axs[0][example_i].imshow(np.reshape(
                test_xs[example_i, :], (height, width, 3)))
            axs[1][example_i].imshow(np.reshape(
                recon[example_i, :], (height, width, 3)))
            axs[2][example_i].imshow(np.reshape(
                test_ys[example_i, :], (height, width, 3)))
            mypath = "C:/Users/Saurabh/Desktop/Tensorflow-101/notebooks/genimages/"
            mypath2 = "C:/Users/Saurabh/Desktop/Tensorflow-101/notebooks/genimages/train/"
            print("Now saving image")
            #imsave(mypath2+"%d.jpg" % (epoch_i), np.reshape(test_xs[example_i,:], (height, width, 3)))
            imsave(mypath+"%d.jpg" % (epoch_i), np.reshape(recon[example_i,:], (height, width, 3)))
        plt.show()
print("Training done. ")    




