class Network_parameters:
    batch_size = 32
    learning_rate = 0.001
    Q_net_var_scope = 'Q_net'
    target_Q_net_var_scope = 'Q_target'
    update_smoothing = 0.9

class Replay_parameters:
    capacity = 50000
    burn_in_episodes = 10000

class Training_parameters:
    discount = 0.99
    test_episodes = 100
    test_frequency = 100
    render_frequency = 1000
    update_target_frequency = 100
    model_save_frequency = 1000
    inital_eps = 0.05
    final_eps = 0.005
    scale_eps = 1

class Directories:
    saved_models = 'models/'
    output = 'output/'