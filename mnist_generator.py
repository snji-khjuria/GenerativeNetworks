from tensorflow.examples.tutorials.mnist import input_data
class MNISTDataGenerator:
    def __init__(self):
        self.mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    def give_next_batch_images(self, batch_size):
        batch_x, _ = self.mnist.train.next_batch(batch_size)
        return batch_x