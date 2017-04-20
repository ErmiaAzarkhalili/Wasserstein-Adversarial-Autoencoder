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

input_dim = dataset.shape[2]*dataset.shape[3]
char_dim = dataset.shape[1]
font_dim = dataset.shape[0]

batch_size = 100
train_iter = int(font_dim*char_dim/batch_size)

hidden_dim_1 = 5000
hidden_dim_2 = 5000

latent_dim = 100
latent_stdev = 5
num_images_per_dim = 25
num_epochs = 50000
decay_epochs = [5000, 25000]
decay_step = np.multiply(train_iter,decay_epochs)
init_learning_rate = 3e-4
decay_rate = .5
lambduh = 10
ndisc = 5

path = None
#path = "../temp/model.cpkt"

class Model():
    def __init__(self, sess, data, nEpochs, init_learning_rate, lambduh, ndisc):
        self.sess = sess
        self.data = data
        self.nEpochs = nEpochs
        self.init_learning_rate = init_learning_rate
        self.lambduh = lambduh
        self.ndisc = ndisc
        self.build_model()
    
    def build_model(self):
        self.x = tf.placeholder(tf.float32, shape=[None,input_dim])
        self.labels = tf.placeholder(tf.float32, shape=[None,char_dim])
        self.sample = tf.placeholder(tf.float32, shape=[1,latent_dim])

        with slim.arg_scope([slim.fully_connected],
                      weights_initializer=tf.random_normal_initializer(stddev=0.01),
                            reuse=True):
            self.enc = slim.fully_connected(self.x, hidden_dim_1, scope='enc1')
            self.enc = slim.fully_connected(self.enc, hidden_dim_2, scope='enc2')
            self.latent = slim.fully_connected(self.enc, latent_dim, activation_fn=None, scope='enc3')
            self.dec = slim.fully_connected(self.latent, hidden_dim_2, scope='dec1') + slim.fully_connected(self.labels, hidden_dim_2, scope='dec2')
            self.dec = slim.fully_connected(self.dec, hidden_dim_1, scope='dec3')
            self.dec = slim.fully_connected(self.dec, input_dim, activation_fn=tf.nn.sigmoid, scope='dec4')
            self.disc_noise = tf.random_normal([tf.shape(self.x)[0],latent_dim])*latent_stdev
            self.disc = slim.fully_connected(self.latent, hidden_dim_2, scope='disc1')
            self.disc = slim.fully_connected(self.disc, hidden_dim_1, scope='disc2')
            self.disc = slim.fully_connected(self.disc, 1, activation_fn=None, scope='disc3')
            self.noise = slim.fully_connected(self.disc_noise, hidden_dim_2, scope='disc1')
            self.noise = slim.fully_connected(self.noise, hidden_dim_1, scope='disc2')
            self.noise = slim.fully_connected(self.noise, 1, activation_fn=None, scope='disc3')
            self.alpha = tf.random_uniform(shape=[tf.shape(self.x)[0],1], minval=0.,maxval=1.)
            self.difference = self.latent - self.disc_noise
            self.interpolates = self.disc_noise + (self.alpha*self.difference)
            self.interp_disc = slim.fully_connected(self.interpolates, hidden_dim_2, scope='disc1')
            self.interp_disc = slim.fully_connected(self.interp_disc, hidden_dim_1, scope='disc2')
            self.interp_disc = slim.fully_connected(self.interp_disc, 1, activation_fn=None, scope='disc3')
            self.gen = slim.fully_connected(self.sample, hidden_dim_2, scope='dec1') + slim.fully_connected(self.labels, hidden_dim_2, scope='dec2')
            self.gen = slim.fully_connected(self.gen, hidden_dim_1, scope='dec3')
            self.gen = slim.fully_connected(self.gen, input_dim, activation_fn=tf.nn.sigmoid, scope='dec4')         

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
        else:
            self.saver.restore(self.sess, path)
            print('Model loaded.')
        sys.stdout.flush()

    def train_iter(self, x, y, epoch):
        _ = self.sess.run([self.rec_enc], feed_dict={self.x : x, self.labels : y, self.sample : np.zeros([1,latent_dim])})
        MSE, _ = self.sess.run([self.MSE, self.rec_dec], feed_dict={self.x : x, self.labels : y, self.sample : np.zeros([1,latent_dim])})
        for _ in range(self.ndisc):
            disc_loss, _ = self.sess.run([self.disc_loss, self.reg_disc], feed_dict={self.x : x, self.labels : y, self.sample : np.zeros([1,latent_dim])})
        gen_loss, _ = self.sess.run([self.gen_loss, self.reg_gen], feed_dict={self.x : x, self.labels : y, self.sample : np.zeros([1,latent_dim])})
        if (self.niter % 100 == 0):
            print('MSE: {}, disc_loss: {},  gen_loss: {}, epoch: {}'.format(MSE,disc_loss,gen_loss,epoch))
        if (self.niter % 10 == 0):
            self.saver.save(self.sess, "../temp/model.cpkt")
            image_capture(epoch, self.niter)
        self.niter = self.niter + 1
        sys.stdout.flush()

    def train(self):
        for i in range(self.nEpochs):
            self.niter = 1
            for x, y in self.data():
                self.train_iter(x, y, i)
            image_capture_fon(epoch)
            if (i % 5 == 0):
                gif_capture(epoch)
                
    def infer(self, x, y, z, gen=False):
        if gen:
            return self.sess.run(self.gen, feed_dict={self.x : x, self.labels : y, self.sample : z})
        else:
            return self.sess.run(self.dec, feed_dict={self.x : x, self.labels : y, self.sample : z})

def data():
    points = np.random.permutation(font_dim*char_dim)
    for offset in range(0, len(points), batch_size):
        s = min(batch_size, len(points) - offset)
        batch = np.zeros((s, input_dim), dtype=np.float32)
        label = np.zeros((s, char_dim), dtype=np.float32)
        for z in range(s):
            point = points[offset + z]
            batch[z] = dataset[int(np.floor(point/char_dim))][point % char_dim].flatten()*1/255
            label[z][point % char_dim] = 1
        yield batch, label

def image_capture(epoch, niter):
    images = []
    for j in range(64):
        thingkern = np.random.normal(size=[1,latent_dim])*latent_stdev
        thinglabel = np.zeros([1,char_dim])
        thinglabel[0,j % 62] = 1
        images.append(model.infer(np.zeros([1,input_dim]),thinglabel,thingkern,gen=True))
    images = np.concatenate(images)
    h, c = plt.subplots(8, 8, figsize=(10, 10))
    h.subplots_adjust(wspace=0,hspace=0)
    for i in range(64):
        c[int(np.floor(i/8))][i % 8].imshow(np.reshape(1-images[i], (dataset.shape[2], dataset.shape[3])), cmap=plt.get_cmap('gray'))
        c[int(np.floor(i/8))][i % 8].axis('off')
    savename = str('../images/rand/fonts' + repr(epoch) + '_' + repr(niter) + '.png')
    h.savefig(savename, format='png', bbox_inches='tight', pad_inches=0, dpi=50)

def image_capture_fon(epoch):    
    for i in range(10):
        images = []
        h, c = plt.subplots(8, 8, figsize=(10, 10))
        h.subplots_adjust(wspace=0,hspace=0)
        for i in range(64):
            c[int(np.floor(i/8))][i % 8].axis('off')
        thingkern = np.random.normal(size=[1,latent_dim])*latent_stdev
        for j in range(62):
            thinglabel = np.zeros([1,char_dim])
            thinglabel[0,j] = 1
            images.append(model.infer(np.zeros([1,input_dim]),thinglabel,thingkern,gen=True))
        images = np.concatenate(images)
        for i in range(62):
            c[int(np.floor(i/8))][i % 8].imshow(np.reshape(1-images[i], (dataset.shape[2], dataset.shape[3])), cmap=plt.get_cmap('gray'))
        savename = str('../images/full/fonts' + repr(epoch) + '_' + repr(i) + '.png')
        h.savefig(savename, format='png', bbox_inches='tight', pad_inches=0, dpi=50)

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
            images.append(model.infer(np.zeros([1,input_dim]),thinglabel,thingkern,gen=True))
        images = np.concatenate(images)
        for i in range(char_dim):
            c[int(np.floor(i/8))][i % 8].imshow(np.reshape(1-images[i], (dataset.shape[2], dataset.shape[3])), cmap=plt.get_cmap('gray'))
        savename = str('../temp/' + repr(k) + '.png')
        h.savefig(savename, format='png', bbox_inches='tight', pad_inches=0, dpi=50)
    gifs = []
    for k in range(gif_len):
        readname = str('../temp/' + repr(k) + '.png')
        gifs.append(imageio.imread(readname))
    imageio.mimsave('../gifs/fonts' + repr(epoch) + '.gif', gifs)        

tf.set_random_seed(13223)
sess = tf.Session()
model = Model(sess, data, num_epochs, init_learning_rate, lambduh, ndisc)
model.train_init(path)
model.train()

for j in range(1):
    out = []
    label = np.zeros((char_dim, char_dim), dtype=np.float32)
    for i in range(char_dim):
        label[i][i] = 1
        out.append(model.infer(np.reshape(dataset[j][i].flatten()*1/255,(1,input_dim)),np.reshape(label[i],(1,char_dim)),np.zeros([1,latent_dim]),gen=False))
    out = np.concatenate(out/np.max(out))
    f, a = plt.subplots(8, 8, figsize=(10, 10))
    g, b = plt.subplots(8, 8, figsize=(10, 10))
    for i in range(char_dim):
        b[int(np.floor(i/8))][i % 8].imshow(np.reshape(1-dataset[j][i]*1/255, (dataset.shape[2], dataset.shape[3])), cmap=plt.get_cmap('gray'))
        a[int(np.floor(i/8))][i % 8].imshow(np.reshape(1-out[i], (dataset.shape[2], dataset.shape[3])), cmap=plt.get_cmap('gray'))
    f.show()
    g.show()
