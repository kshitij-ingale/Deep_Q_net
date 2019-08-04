# Deep_Q_net
Implementation for Deep Q networks using Tensorflow 2.0 beta. Extensions to DQN like double DQN and Dueling networks are supported
## Run instructions
The following command can be executed for training agent on the implementation for default Gym environment of CartPole. 
```
python agent.py
```
In order to render environment, --r argument can be provided which renders environment as per frequency as indicated in config.py. In order to use double DQN or dueling networks, arguments db or du can be specified