import tensorflow as tf

#discrminator function
def discriminator(x, parameters):
    with tf.name_scope('discriminator_model') as scope:
        weights = parameters["weights"]
        biases = parameters["biases"]
        hidden = tf.nn.relu(tf.add(tf.matmul(x, weights["disc_hidden"]), biases["disc_hidden"]), name='hidden')
        out    = tf.nn.sigmoid(tf.add(tf.matmul(hidden, weights["disc_out"]), biases["disc_out"]), name='out')
    return out


#generator function
def generator(x, parameters):
    with tf.name_scope('generator_model') as scope:
        weights = parameters["weights"]
        biases = parameters["biases"]
        hidden = tf.nn.relu(tf.add(tf.matmul(x, weights["gen_hidden"]), biases["gen_hidden"]), name='hidden')
        out    = tf.nn.sigmoid(tf.add(tf.matmul(hidden, weights["gen_out"]), biases["gen_out"]), name='out')
    return out


#xavier initialization
def xavier_init(shape):
    return tf.random_normal(shape=shape, stddev=1./tf.sqrt(shape[0]/2.))


#give me parameters
def give_me_parameters(config):
    noise_dim       = config.noise_dim
    gen_hidden_dim  = config.gen_hidden_dim
    image_dim       = config.image_dim
    disc_hidden_dim = config.disc_hidden_dim
    with tf.name_scope('disc_weights') as scope:
        w_disc_hidden = tf.Variable(xavier_init(shape=[image_dim, disc_hidden_dim]), name='w_disc_hidden')
        w_disc_out = tf.Variable(xavier_init(shape=(disc_hidden_dim, 1)), name='w_disc_out')
        b_disc_hidden = tf.Variable(tf.zeros([disc_hidden_dim]), name='b_disc_hidden')
        b_disc_out = tf.Variable(tf.zeros([1]), name='b_disc_out')
    with tf.name_scope('gen_weights') as scope:
        w_gen_hidden = tf.Variable(xavier_init(shape=[noise_dim, gen_hidden_dim]), name='w_gen_hidden')
        w_gen_out = tf.Variable(xavier_init(shape=[gen_hidden_dim, image_dim]), name='w_gen_out')
        b_gen_hidden = tf.Variable(tf.zeros([gen_hidden_dim]), name='b_gen_hidden')
        b_gen_out = tf.Variable(tf.zeros([image_dim]), name='b_gen_out')
    weights = {
        'gen_hidden' : w_gen_hidden,
        'gen_out'    : w_gen_out,
        'disc_hidden': w_disc_hidden,
        'disc_out'   : w_disc_out
    }
    biases = {
        'gen_hidden' : b_gen_hidden,
        'gen_out'    : b_gen_out,
        'disc_hidden': b_disc_hidden,
        'disc_out'   : b_disc_out
    }
    parameters = {}
    parameters["weights"] = weights
    parameters["biases"]  = biases
    return parameters

def give_vars(parameters, var_list):
    vars = []
    w = parameters["weights"]
    b = parameters["biases"]
    for v in var_list:
        vars.append(w[v])
        vars.append(b[v])
    return vars


def give_gen_vars(parameters):
    var_list = ["gen_hidden", "gen_out"]
    return give_vars(parameters, var_list)


def give_disc_vars(parameters):
    var_list = ["disc_hidden", "disc_out"]
    return give_vars(parameters, var_list)
