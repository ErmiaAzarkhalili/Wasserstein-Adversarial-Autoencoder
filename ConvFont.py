#JAERYUNG SONG FINAL

import sys
import h5py
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio

f = h5py.File('../../fonts.hdf5', 'r')
dataset = f['fonts']

input_dim_1 = dataset.shape[2]
input_dim_2 = dataset.shape[3]
input_dim = input_dim_1*input_dim_2
char_dim = dataset.shape[1]
font_dim = dataset.shape[0]
channels = 1

batch_size = 100
train_iter = int(font_dim*char_dim/batch_size)

hidden_dim_1 = 16
hidden_dim_2 = 32
kernel = [9, 9]
stride = 1
reconst_dim_1 = input_dim_1-(kernel[0]-1)*4
reconst_dim_2 = input_dim_2-(kernel[1]-1)*4
reconst_dim = reconst_dim_1*reconst_dim_2*hidden_dim_2

latent_dim = 100
latent_stdev = 20
fc_dim = latent_dim*10
num_epochs = 50000
decay_epochs = [100, 10000]
decay_step = np.multiply(train_iter,decay_epochs)
init_learning_rate = 3e-4
decay_rate = .5
lambduh = 10
ndisc = 5

path = None
#path = "../temp/conv_model.cpkt"

class Model():
    def __init__(self, sess, data, nEpochs, init_learning_rate, lambduh, ndisc):
        self.sess = sess
        self.data = data
        self.nEpochs = nEpochs
        self.init_learning_rate = init_learning_rate
        self.lambduh = lambduh
        self.ndisc = ndisc
        self.build_model()
    
    def Encoder(self, inputs):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_initializer=tf.random_normal_initializer(stddev=0.01),
                            reuse=True):
            with slim.arg_scope([slim.conv2d], kernel_size=kernel, stride=stride, padding='VALID'):
                output = slim.conv2d(inputs, hidden_dim_1, scope='enc1')
                output = slim.conv2d(output, hidden_dim_2, scope='enc4')
                output = tf.reshape(output, [-1, reconst_dim*channels])
                output = slim.fully_connected(output, fc_dim, scope='enc5')
                output = slim.fully_connected(output, latent_dim, activation_fn=None, scope='enc6')
        return output
    
    def Decoder(self, inputs, labels):
        with slim.arg_scope([slim.convolution2d_transpose, slim.fully_connected],
                            weights_initializer=tf.random_normal_initializer(stddev=0.01),
                            reuse=True):
            with slim.arg_scope([slim.convolution2d_transpose], kernel_size=kernel, stride=stride, padding='VALID'):
                output = slim.fully_connected(inputs, fc_dim, scope='dec1') + slim.fully_connected(labels, fc_dim, scope='dec2')
                output = slim.fully_connected(output, reconst_dim, scope='dec3')
                output = tf.reshape(output, [-1, reconst_dim_1, reconst_dim_2, hidden_dim_2])
                output = slim.convolution2d_transpose(output, hidden_dim_1, scope='dec6')
                output = slim.convolution2d_transpose(output, channels, scope='dec7')
        return output
    
    def Discriminator(self, inputs):
        with slim.arg_scope([slim.convolution2d_transpose, slim.fully_connected],
                            weights_initializer=tf.random_normal_initializer(stddev=0.01),
                            reuse=True):
            with slim.arg_scope([slim.convolution2d_transpose], kernel_size=kernel, stride=stride, padding='VALID'):
                output = slim.fully_connected(inputs, fc_dim, scope='disc1')
                output = slim.fully_connected(output, reconst_dim, scope='disc2')
                output = tf.reshape(output, [-1, reconst_dim_1, reconst_dim_2, hidden_dim_2])
                output = slim.convolution2d_transpose(output, hidden_dim_1, scope='disc5')
                output = slim.convolution2d_transpose(output, channels, scope='disc6')
                output = tf.reshape(output, [-1, input_dim*channels])
                output = slim.fully_connected(output, 1, activation_fn=None, scope='disc7')
        return output
    
    def build_model(self):
        self.x = tf.placeholder(tf.float32, shape=[None,input_dim_1,input_dim_2,channels])
        self.labels = tf.placeholder(tf.float32, shape=[None,char_dim])
        self.sample = tf.placeholder(tf.float32, shape=[1,latent_dim])

        self.latent = self.Encoder(self.x)
        self.dec = self.Decoder(self.latent, self.labels)
        self.gen = self.Decoder(self.sample, self.labels)
        
        self.disc_noise = tf.random_normal([tf.shape(self.x)[0],latent_dim])*latent_stdev
        self.disc = self.Discriminator(self.latent)
        self.noise = self.Discriminator(self.disc_noise)

        self.alpha = tf.random_uniform(shape=[tf.shape(self.x)[0],1], minval=0.,maxval=1.)
        self.difference = self.latent - self.disc_noise
        self.interpolates = self.disc_noise + (self.alpha*self.difference)
        
        self.interp_disc = self.Discriminator(self.interpolates)     

        self.MSE = tf.losses.mean_squared_error(self.x, self.dec)
        self.disc_loss = tf.reduce_mean(self.disc)
        self.disc_loss += -tf.reduce_mean(self.noise)
        self.gradients = tf.gradients(self.interp_disc, [self.interpolates])[0]
        self.slopes = tf.sqrt(tf.reduce_sum(tf.square(self.gradients), reduction_indices=[1]))
        self.gradient_penalty = tf.reduce_mean((self.slopes-1.)**2)
        self.disc_loss += self.lambduh*self.gradient_penalty
        self.gen_loss = -tf.reduce_mean(self.disc)
    
    def train_init(self, path):
        enc_variables = slim.get_model_variables('enc')
        dec_variables = slim.get_model_variables('dec')
        disc_variables = slim.get_model_variables('disc')
        global_step = tf.Variable(0, trainable=False)
        
        def f0():
            return tf.multiply(decay_rate*decay_rate,self.init_learning_rate)
        def f1():
            return tf.train.exponential_decay(decay_rate*self.init_learning_rate, global_step, decay_step[1], decay_rate, staircase=True)
        def f2():
            return tf.train.exponential_decay(self.init_learning_rate, global_step, decay_step[0], decay_rate, staircase=True)
        
        pred = tf.greater(global_step,tf.Variable(decay_step[0], dtype=tf.int32))
        pred1 = tf.greater(global_step,tf.Variable(decay_step[1], dtype=tf.int32))
        self.learning_rate = tf.cond(pred, lambda: tf.cond(pred1, f0, f1), f2)

        self.saver = tf.train.Saver()
            
        self.rec_enc = tf.train.AdamOptimizer(self.learning_rate, .9, .9, name='Rec_enc').minimize(self.MSE, var_list=enc_variables)
        self.rec_dec = tf.train.AdamOptimizer(self.learning_rate, .9, .9, name='Rec_dec').minimize(self.MSE, var_list=dec_variables),
        self.reg_disc = tf.train.AdamOptimizer(self.learning_rate*10, .1, .9, name='Reg_disc').minimize(self.disc_loss, var_list=disc_variables),
        self.reg_gen = tf.train.AdamOptimizer(self.learning_rate*10, .1, .9, name='Reg_gen').minimize(self.gen_loss, var_list=enc_variables, global_step=global_step)

        if not path:
            self.sess.run(tf.global_variables_initializer())
            print('Initialization complete.')
        elif isinstance(path, str):
            self.saver.restore(self.sess, path)
            print('Model loaded.')
        sys.stdout.flush()

    def train_iter(self, x, y, epoch):
        _ = self.sess.run([self.rec_enc], feed_dict={self.x : x, self.labels : y, self.sample : np.zeros([1,latent_dim])})
        MSE, _ = self.sess.run([self.MSE, self.rec_dec], feed_dict={self.x : x, self.labels : y, self.sample : np.zeros([1,latent_dim])})
        for _ in range(self.ndisc):
            disc_loss, _ = self.sess.run([self.disc_loss, self.reg_disc], feed_dict={self.x : x, self.labels : y, self.sample : np.zeros([1,latent_dim])})
        gen_loss, _ = self.sess.run([self.gen_loss, self.reg_gen], feed_dict={self.x : x, self.labels : y, self.sample : np.zeros([1,latent_dim])})
        if (self.niter % 1 == 0):
            print('MSE: {}, disc_loss: {},  gen_loss: {}, epoch: {}'.format(MSE,disc_loss,gen_loss,epoch))
        if (self.niter % 1000 == 0):
            self.saver.save(self.sess, "../temp/conv_model.cpkt")
            image_capture(epoch, self.niter)
        self.niter = self.niter + 1
        sys.stdout.flush()

    def train(self):
        for i in range(self.nEpochs):
            self.niter = 1
            for x, y in self.data():
                self.train_iter(x, y, i)
            image_capture_fon(i)
            if (i % 5 == 1):
                gif_capture(i)
                
    def infer(self, x, y, z, gen=False):
        if gen:
            return self.sess.run(self.gen, feed_dict={self.x : x, self.labels : y, self.sample : z})
        else:
            return self.sess.run(self.dec, feed_dict={self.x : x, self.labels : y, self.sample : z})

def data():
    points = np.random.permutation(font_dim*char_dim)
    for offset in range(0, len(points), batch_size):
        s = min(batch_size, len(points) - offset)
        batch = np.zeros((s, input_dim_1, input_dim_2, channels), dtype=np.float32)
        label = np.zeros((s, char_dim), dtype=np.float32)
        for z in range(s):
            point = points[offset + z]
            batch[z] = np.reshape(dataset[int(np.floor(point/char_dim))][point % char_dim]*1/255, [input_dim_1, input_dim_2, channels])
            label[z][point % char_dim] = 1
        yield batch, label

def image_capture(epoch, niter):
    images = []
    for j in range(64):
        thingkern = np.random.normal(size=[1,latent_dim])*latent_stdev
        thinglabel = np.zeros([1,char_dim])
        thinglabel[0,j % char_dim] = 1
        images.append(model.infer(np.zeros([1,input_dim_1,input_dim_2,channels]),thinglabel,thingkern,gen=True))
    images = np.concatenate(images)
    h, c = plt.subplots(8, 8, figsize=(10, 10))
    h.subplots_adjust(wspace=0,hspace=0)
    for i in range(64):
        c[int(np.floor(i/8))][i % 8].imshow(np.reshape(1-images[i],[input_dim_1,input_dim_2]), cmap=plt.get_cmap('gray'))
        c[int(np.floor(i/8))][i % 8].axis('off')
    savename = str('../images/rand/conv/fonts' + repr(epoch) + '_' + repr(niter) + '.png')
    h.savefig(savename, format='png', bbox_inches='tight', pad_inches=0, dpi=50)
    plt.close("all")

def image_capture_fon(epoch):    
    for i in range(10):
        images = []
        h, c = plt.subplots(8, 8, figsize=(10, 10))
        h.subplots_adjust(wspace=0,hspace=0)
        for i in range(64):
            c[int(np.floor(i/8))][i % 8].axis('off')
        thingkern = np.random.normal(size=[1,latent_dim])*latent_stdev
        for j in range(char_dim):
            thinglabel = np.zeros([1,char_dim])
            thinglabel[0,j] = 1
            images.append(model.infer(np.zeros([1,input_dim_1,input_dim_2,channels]),thinglabel,thingkern,gen=True))
        images = np.concatenate(images)
        for i in range(char_dim):
            c[int(np.floor(i/8))][i % 8].imshow(np.reshape(1-images[i],[input_dim_1,input_dim_2]), cmap=plt.get_cmap('gray'))
        savename = str('../images/full/conv/fonts' + repr(epoch) + '_' + repr(i) + '.png')
        h.savefig(savename, format='png', bbox_inches='tight', pad_inches=0, dpi=50)
    plt.close("all")

def gif_capture(epoch):
    gif_len = 100
    thingkern0 = np.random.normal(size=[1,latent_dim])*latent_stdev
    thingkern1 = np.random.normal(size=[1,latent_dim])*latent_stdev
    for k in range(gif_len):
        images = []
        if (k % 15) == 0:
            h, c = plt.subplots(8, 8, figsize=(10, 10))
            h.subplots_adjust(wspace=0,hspace=0)
            for i in range(64):
                c[int(np.floor(i/8))][i % 8].axis('off')
        thingkern = thingkern0 + (thingkern1 - thingkern0)*k/(gif_len-1)
        for j in range(char_dim):
            thinglabel = np.zeros([1,char_dim])
            thinglabel[0,j] = 1
            images.append(model.infer(np.zeros([1,input_dim_1,input_dim_2,channels]),thinglabel,thingkern,gen=True))
        images = np.concatenate(images)
        for i in range(char_dim):
            c[int(np.floor(i/8))][i % 8].imshow(np.reshape(1-images[i],[input_dim_1,input_dim_2]), cmap=plt.get_cmap('gray'))
        savename = str('../temp/conv/' + repr(k) + '.png')
        h.savefig(savename, format='png', bbox_inches='tight', pad_inches=0, dpi=50)
    gifs = []
    for k in range(gif_len):
        readname = str('../temp/conv/' + repr(k) + '.png')
        gifs.append(imageio.imread(readname))
    imageio.mimsave('../gifs/conv/fonts' + repr(epoch) + '.gif', gifs)        

tf.set_random_seed(13223)
sess = tf.Session()
model = Model(sess, data, num_epochs, init_learning_rate, lambduh, ndisc)
model.train_init(path)
if not path:
    model.train()

def reconstructor(font_num):
    out = []
    label = np.zeros((char_dim, char_dim), dtype=np.float32)
    for i in range(char_dim):
        label[i][i] = 1
        out.append(model.infer(dataset[font_num][i]*1/255,np.reshape(label[i],(1,char_dim)),np.zeros([1,latent_dim]),gen=False))
    out = np.concatenate(out/np.max(out))
    f, a = plt.subplots(8, 8, figsize=(10, 10))
    g, b = plt.subplots(8, 8, figsize=(10, 10))
    for i in range(char_dim):
        b[int(np.floor(i/8))][i % 8].imshow(1-dataset[font_num][i]*1/255, cmap=plt.get_cmap('gray'))
        a[int(np.floor(i/8))][i % 8].imshow(1-out[i], cmap=plt.get_cmap('gray'))
    f.show()
    g.show()
