import tensorflow as tf

from config import MnistConfig
from mnist_generator import MNISTDataGenerator
from mnist_gan import MnistGAN
from mnist_trainer import MnistTrainer
print("Building Mnist GAN Model...")
model = MnistGAN(MnistConfig)
print("Model is available")
sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)
print("Initializing data generator...")
data_generator = MNISTDataGenerator()
print("Data Generator Available")
import tensorflow as tf
logs_location = MnistConfig.tensorboard_logs
# writer.add_graph(sess.graph)
print("Making the training process....")
trainer = MnistTrainer(sess, MnistConfig, data_generator, logs_location)
print("Starting the training of the model")
trainer.train_model(model)
print("Model training done.")
sess.close()