class MnistConfig:
    #training parameters
    num_steps     = 70000
    batch_size    = 128
    learning_rate = 0.0002

    #network parameters
    image_dim       = 784
    gen_hidden_dim  = 256
    disc_hidden_dim = 256
    noise_dim       = 100

    tensorboard_logs = '/home/maulik/Desktop/logs'