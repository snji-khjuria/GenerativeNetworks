import numpy as np
import tensorflow as tf
class MnistTrainer:
    def __init__(self, sess, config, data_generator, logs_location):
        self.sess = sess
        self.config = config
        self.data_generator = data_generator
        self.file_writer = tf.summary.FileWriter(logs_location, sess.graph)
        # self.file_writer = file_writer


    def train_model(self, model):
        num_steps = self.config.num_steps
        batch_size = self.config.batch_size
        noise_dim  = self.config.noise_dim
        for i in range(num_steps):
            #get real input
            batch_x = self.data_generator.give_next_batch_images(batch_size)
            #generate noise to be fed into generator
            z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])
            gen_loss, disc_loss, summary_out = model.train_model(self.sess, batch_x, z)
            if i%200==0:
                print("Step %i: Generator loss: %f, Discriminator Loss: %f" % (i, gen_loss, disc_loss))
                self.file_writer.add_summary(summary_out, i)
                self.file_writer.flush()