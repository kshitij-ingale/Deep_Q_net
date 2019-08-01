class Network_parameters:
    layers=[16,32]
    activation='relu'
    batch_size = 32
    learning_rate = 0.005
    Q_net_var_scope = 'Q_net'
    target_Q_net_var_scope = 'Q_target'

class Replay_parameters:
    capacity = 50000
    burn_in_episodes = 10000

class Training_parameters:
    discount = 0.999
    test_episodes = 100
    test_frequency = 100
    render_frequency = 3333
    update_target_frequency = 100
    model_save_frequency = 1000

class Directories:
    tensorboard = 'tensorboard/'