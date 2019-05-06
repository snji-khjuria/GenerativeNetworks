import tensorflow as tf
from utils import generator, give_me_parameters, discriminator, give_gen_vars, give_disc_vars
class MnistGAN:

    def __init__(self, config):
        self.config = config
        self.build_model()


    def build_model(self):
        noise_dim = self.config.noise_dim
        image_dim = self.config.image_dim
        learning_rate = self.config.learning_rate
        #create placeholders
        self.gen_input = tf.placeholder(tf.float32, shape=[None, noise_dim], name="input_noise")
        self.disc_input = tf.placeholder(tf.float32, shape=[None, image_dim], name="disc_input")

        self.parameters = give_me_parameters(self.config)
        #build generator network
        with tf.name_scope('generator') as scope:
            self.gen_sample = generator(self.gen_input, self.parameters)
        with tf.name_scope('discriminator') as scope:
            disc_real           = discriminator(self.disc_input, self.parameters)
            disc_fake           = discriminator(self.gen_sample, self.parameters)

        with tf.name_scope('loss') as scope:
            #make losses
            self.gen_loss = -tf.reduce_mean(tf.log(disc_fake))
            self.disc_loss = -tf.reduce_mean(tf.log(disc_real) + tf.log(1. - disc_fake))

        with tf.name_scope('summaries') as scope:
            gen_images = tf.reshape(self.gen_sample, [-1, 28, 28, 1])
            tf.summary.scalar('Generative Loss', self.gen_loss)
            tf.summary.scalar('Discriminator Loss', self.disc_loss)
            #tf.summary.image("Generated image", gen_images)
            tf.summary.image("Generated image", gen_images, 100, family='generated_images')
            self.merged_summary = tf.summary.merge_all()
        with tf.name_scope('optimizers') as scope:
            #create Optimizers
            optimizer_gen  = tf.train.AdamOptimizer(learning_rate=learning_rate)
            optimizer_disc = tf.train.AdamOptimizer(learning_rate=learning_rate)
            gen_vars  = give_gen_vars(self.parameters)
            disc_vars = give_disc_vars(self.parameters)
            self.train_gen = optimizer_gen.minimize(self.gen_loss, var_list=gen_vars)
            self.train_disc = optimizer_disc.minimize(self.disc_loss, var_list=disc_vars)


    def train_model(self, sess, images, noise):
        feed_dict = {self.gen_input:noise, self.disc_input:images}
        _, _, gl_loss_out, disc_loss_out, summary_out = sess.run([self.train_gen, self.train_disc, self.gen_loss, self.disc_loss, self.merged_summary], feed_dict=feed_dict)
        return gl_loss_out, disc_loss_out, summary_out