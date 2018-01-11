import os
import numpy as np
import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import skimage
from load import load
from srgan_imagenet import SRGAN

learning_rate =1e-4
d_learning_rate  =1e-3
batch_size = 16 
img_dim = 96

vgg_model = '../vgg19/backup/latest'

def train():
   # set.seed(50)
    model = SRGAN()
    tf.summary.scalar("generatorloss", model.g_loss)
    tf.summary.scalar("discrimantorloss", model.d_loss)
   # tf.summary.scalar("D loss Fake",model.d_loss_fake)
   # tf.summary.scalar("D loss Real",model.d_loss_real)
    #tf.summary.:wqscalar("contentloss",content_loss)
   # tf.summary.image("GEN SR TEST",fake)
   # tf.summary.image("LR TEST", mos)
    merged_summary_op = tf.summary.merge_all()
    sess = tf.Session()
    with tf.variable_scope('srgan'):
        global_step = tf.Variable(0, name='global_step', trainable=False)
       # d_global_step = int(sess.run(global_step)/n_iter+1 /2)
   # print("Global step is:",global_step)
   # tf.Print(global_step,[global_step], message="Global Step:")    
    g_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(model.g_loss, global_step=global_step, var_list=model.g_variables)
   # d_train_op = tf.train.AdamOptimizer(learning_rate=d_learning_rate).minimize(model.d_loss, var_list=model.d_variables)
    d_train_op = tf.train.AdamOptimizer(learning_rate = d_learning_rate).minimize(model.d_loss, global_step = global_step, var_list = model.d_variables) 
   # d_train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(model.d_loss, global_step=global_step, var_list=model.d_variables)
    d_train_op_adam = tf.train.AdamOptimizer(learning_rate = learning_rate)
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter('/tmp/tensorflow_logs_with_imagenet_cross', graph=sess.graph)
    var_ = tf.global_variables()

    # Restore the VGG-19 network
    vgg_var = [var for var in var_ if "vgg19" in var.name]
    saver = tf.train.Saver(vgg_var)
    saver.restore(sess, vgg_model)

    # Restore the SRGAN network
    if tf.train.get_checkpoint_state('backup/'):
        saver = tf.train.Saver()
        saver.restore(sess, 'backup/latest')

    # Loading the data
    x_train = load('cyto/train')
    x_test = load('cyto/test')
   # x_train = load('dementer/users/saurabh/lfw/train')
   # x_test = load('dementer/users/saurabh/lfw/test')

    # Train the SRGAN model
    n_iter = int(np.ceil(len(x_train) / batch_size))
    print(len(x_train))   
    print('NO of iterations:',n_iter)
    while True:
        epoch = int(sess.run(global_step) / n_iter+1 / 2)
        print('epoch: {}'.format(epoch + 1))
        #epoch = 350
        perm = np.random.permutation(len(x_train))
        x_train = x_train[perm]
        perm2 = np.random.permutation(len(x_test))
        x_test = x_test[perm2]
        
       # print("Epoch:",epoch_i)
        for i in tqdm(range(n_iter)):
            x_batch = x_train[i * batch_size:(i + 1) * batch_size]
           # x_batch = x_test[i*batch+size:(i+1)*batch_size]
            opt, summary = sess.run([g_train_op,merged_summary_op], feed_dict={model.x: x_batch, model.is_training: True})
            #summary_writer.add_summary(summary, epoch_i * total_batch + batch_i) 
           # tf.summary.image('testhr',sess.run([model.imitation], feed_dict={model.x:x_test[:batch_size],model.is_training:False}) )
           # mos, fake = sess.run([model.downscaled, model.imitation], feed_dict = {model.x:x_test[:batch_size], model.is_training: False})
           # tf.summary.image("Gen SR TEST", fake)
           # tf.summary.image("LR teSt", mos)
           # print(mos)
           # validate(x_test, epoch, model,sess)
           # raw = x_test[:batch_size]
           # mos,fake = sess.run([model.downscaled, model.imitation], feed_dict = {model.x:raw, model.is_training:False})
           # tf.summary.image("GEN HR TEST",fake)
           # tf.summary.image("LR TEST", mos)
            summary_writer.add_summary(summary, epoch * n_iter + i)
           # sess.run([g_train_op, d_train_op], feed_dict={model.x: x_batch, model.is_training: True})
           # saveimg([mos, fake, raw], ['Input test','Output test','Truth test'], epoch)
           # print("Global step is :  ",sess.run(global_step)) 
            
            validate(x_test, epoch, model, sess)
           # mos, fake = sess.run([model.downscaled, model.imitation], feed_dict = {model.x:x_test[:batch_size], model.is_training: False})
           # tf.summary.image("Gen SR TEST", fake)
           # tf.summary.image("LR TEST", mos)
           # print("valid")
        # Validate
          # validate(x_test, epoch, model, sess)

        # Save the model
        saver = tf.train.Saver()
        saver.save(sess, 'backup/latest', write_meta_graph=False)


def validate(x_test, epoch, model, sess):
    raw = x_test[:batch_size]
    mos, fake = sess.run([model.downscaled, model.imitation], feed_dict={model.x: raw, model.is_training: False})
      
    saveimg([mos, fake, raw], ['Input', 'Output', 'Ground Truth'], epoch)


def saveimg(imgs, label, epoch):
    for i in range(batch_size):
        fig = plt.figure()
        for j, img in enumerate(imgs):
           # print("the min is ",img[i].min())
           # print("the max is",img[i].max())
           # im = skimage.img_as_ubyte(img[i], force_copy=False)
           # im = np.uint8((img[i] + 1.5)*127.5 )
           # print("The min is ",im.min())
            img[i] =  np.clip(img[i],-1,1)
            print("The min is",img[i].min())
            im = np.uint8((img[i] + 1)*127.5)
           # im = im32.astype('uint8')        
           # im = tf.cast(img[i], tf.uint8)
           # im = tf.image_convert_image_dtype(img[i], tf.uint8)
           # print(im)
            fig.add_subplot(1, len(imgs), j+1)
            plt.imshow(im)
            plt.tick_params(labelbottom='off')
            plt.tick_params(labelleft='off')
            plt.gca().get_xaxis().set_ticks_position('none')
            plt.gca().get_yaxis().set_ticks_position('none')
            plt.xlabel(label[j])
        seq_ = "{0:09d}".format(i + 1)
        epoch_ = "{0:09d}".format(epoch + 1)
        path = os.path.join('result', seq_, '{}.jpg'.format(epoch_))
        if os.path.exists(os.path.join('result', seq_)) == False:
            os.mkdir(os.path.join('result', seq_))
        plt.savefig(path)
        plt.close()

if __name__ == '__main__':
    train()

