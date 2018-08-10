import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from numpy.random import normal
from util import *
import layers
from glob import glob
import os

class dcgan() :
    def __init__(self,batch_size,noise_size ):
        self.batch_size = batch_size
        self.noise_size = noise_size
        self.training_epoch = 500
        self.path = 'kakao/kakao/*.jpg'

    def generator(self,input,reuse= True):
        depth = [1024,512,256,128,3]

        with tf.variable_scope("Generator", reuse = reuse):
            with tf.variable_scope("g_1",reuse=reuse):
                net = layers.dc_dense(input , 4*4*depth[0],"g_w1")
                net = tf.reshape(net, [-1, 4,4,depth[0]])
                net = tf.nn.relu(net)
            with tf.variable_scope("g_2",reuse= reuse):
                net = tf.nn.relu(layers.batch_norm(layers.dc_deconv(net,depth[1],"g_w2")))
            with tf.variable_scope("g_3",reuse=reuse):
                net = tf.nn.relu(layers.batch_norm(layers.dc_deconv(net,depth[2],"g_w3")))
            with tf.variable_scope("g_4",reuse=reuse):
                net = tf.nn.relu(layers.batch_norm(layers.dc_deconv(net,depth[3],"g_w4")))
            with tf.variable_scope("g_5",reuse=reuse):
                net = layers.dc_deconv(net,depth[4],"g_w5")
                net = tf.nn.tanh(net)

            return net

    def discriminator(self,input, reuse = True):
        depth = [64,128,256,512,1]
        with tf.variable_scope("Discriminator", reuse = reuse):
            with tf.variable_scope("d_1", reuse= reuse):
                net = lrelu(layers.batch_norm(layers.dc_conv(input, depth[0],'d_w1')))
            with tf.variable_scope("d_2", reuse=reuse):
                net = lrelu(layers.batch_norm(layers.dc_conv(net, depth[1],'d_w2')))
            with tf.variable_scope("d_3",reuse=reuse):
                net = lrelu(layers.batch_norm(layers.dc_conv(net, depth[2],'d_w3')))
            with tf.variable_scope("d_4", reuse= reuse):
                net = lrelu(layers.batch_norm(layers.dc_conv(net, depth[3],'d_w4')))
            with tf.variable_scope("d_5", reuse = reuse):
                net = layers.flatten(net)
                net = layers.dc_dense(net, 1,name = "d_fc")

        return net

    def build_model(self):
        self.x = tf.placeholder(tf.float32, shape = [None, 64,64,3])
        self.z = tf.placeholder(tf.float32, shape = [None, self.noise_size])

        self.g = self.generator(self.z, reuse = False)
        self.d_real = self.discriminator(self.x , reuse = False)
        self.d_fake = self.discriminator(self.g, reuse = True)

    def loss_op(self):

        d_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.d_real, labels = tf.ones_like(self.d_real)))
        d_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.d_fake, labels = tf.zeros_like(self.d_fake)))
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.d_fake, labels = tf.ones_like(self.d_fake)))

        self.d_loss = d_real_loss + d_fake_loss

        t_var = tf.trainable_variables()

        d_var = [var for var in t_var if 'd_' in var.name]
        g_var = [var for var in t_var if 'g_' in var.name]

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):

            self.d_trainer = tf.train.AdamOptimizer(learning_rate=0.0002, beta1 = 0.5).minimize(self.d_loss,var_list=d_var)
            self.g_trainer = tf.train.AdamOptimizer(learning_rate=0.0002, beta1 = 0.5).minimize(self.g_loss,var_list=g_var)

    def train(self):
        self.build_model()
        self.loss_op()
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        training_epoch = 500
        with tf.Session() as sess:
            sess.run(init)
            total_batch = int((493)/self.batch_size)-1
            for epoch in range(training_epoch):
                file_list = glob(self.path)

                for iteration in range(total_batch):
                    real_image, file_list = dataload(file_list,batch_size=self.batch_size)
                    fake_image = normal(size = [self.batch_size,self.noise_size])
                    _ , d_loss_val = sess.run([self.d_trainer, self.d_loss], feed_dict= {self.x : real_image, self.z : fake_image})
                    _ , g_loss_val = sess.run([self.g_trainer, self.g_loss], feed_dict ={self.x : real_image, self.z : fake_image})


                    print("epoch  :  {}  ,  iteration  :  {}  ,  d_loss  :  {},  g_loss  :  {}".format(epoch,iteration,d_loss_val, g_loss_val))
                    # sample picture
                    makedir("samples")
                    # save check_dir
                    makedir("check_dir")
                    saver.save(sess,os.path.join("check_dir","kakaogan"),global_step = epoch)



                    if iteration % 10 ==0 or iteration ==0  :
                        sample_size = 10
                        sample_noise = normal(size = [sample_size, self.noise_size])
                        samples = sess.run(self.g , feed_dict={self.z : sample_noise})


                        fig , ax = plt.subplots(1, 10, figsize = (10,10))
                        for j in range(sample_size):
                            ax[j].imshow(samples[j])
                            ax[j].set_axis_off()

                        plt.savefig("samples/{}_{}.jpg".format(str(epoch), str(iteration).zfill(3)), bbox_inches = 'tight')
                        plt.close(fig)

#     def test(self):
#         print("Testing the results")
#         self.build_model()
#         saver = tf.train.Saver()
#         init = tf.global_variables_initializer()

#         with tf.Session() as sess:
#             sess.run()

#             # make test_dir
#             makedir("test")

#             random_noise = normal(size = 10, self.noise_size)
#             sample_image = sess.run()




train = True

def main():
    model = dcgan(10, 200)

    if train :
        model.train()
    elif test:
        model.test()

if __name__ =="__main__":
    main()
