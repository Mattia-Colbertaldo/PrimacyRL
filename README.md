# PrimacyRL
Reset the last layer for exploration in RL.

# Abstract of the original paper
"The Primacy Bias in Deep Reinforcement Learning"

Evgenii Nikishin, Max Schwarzer, Pierluca D’Oro, Pierre-Luc Bacon, Aaron Courville

This work identifies a common flaw of deep reinforcement learning (RL) algorithms: a tendency
to rely on early interactions and ignore useful evidence encountered later. Because of training on
progressively growing datasets, deep RL agents
incur a risk of overfitting to earlier experiences,
negatively affecting the rest of the learning process. Inspired by cognitive science, we refer to
this effect as the primacy bias. Through a series
of experiments, we dissect the algorithmic aspects
of deep RL that exacerbate this bias. We then propose a simple yet generally-applicable mechanism
that tackles the primacy bias by periodically resetting a part of the agent. We apply this mechanism
to algorithms in both discrete (Atari 100k) and
continuous action (DeepMind Control Suite) domains, consistently improving their performance.

"Your assumptions are your windows on the world. Scrub
them off every once in a while, or the light won’t come in.”
–Isaac Asimov

# Useful links
* Dopamine Notebooks: https://github.com/google/dopamine/tree/master/dopamine/colab

# State-of-art result of Dopamine on the training of Cartpole over 100 iterations for 10 times
![Alt text](https://github.com/Mattia-Colbertaldo/PrimacyRL/blob/main/demo.png)

# Enhancing Dopamine with Restart Functionality: A Step-by-Step Guide

# First Step
Since in the Colab Notebook we use:
```ruby
DQNAgent.network = @gym_lib.CartpoleDQNNetwork
```
In https://github.com/Mattia-Colbertaldo/dopamine_restart/blob/master/dopamine/discrete_domains/gym_lib.py, we added "reset_last_layer" method which resets the last fully connected layer by simply creating it.

Code:
In the class
```ruby
class CartpoleDQNNetwork(tf.keras.Model)
```
We added
```ruby
  def reset_last_layer(self):
    self.net.reset_last_layer()
```
Where net is 
```ruby
self.net = BasicDiscreteDomainNetwork(CARTPOLE_MIN_VALS, CARTPOLE_MAX_VALS, num_actions)
```
And in the class
```ruby
class BasicDiscreteDomainNetwork(tf.keras.layers.Layer):
```
We added:
```ruby
  def reset_last_layer(self):
    """Reset the last layer of the network."""
    if self.num_atoms is None:
      self.last_layer = tf.keras.layers.Dense(self.num_actions,
                                              name='fully_connected')
    else:
      self.last_layer = tf.keras.layers.Dense(self.num_actions * self.num_atoms,
                                              name='fully_connected')
```
        
# Second Step     
Change the colab so that clone our forked repo "https://github.com/Mattia-Colbertaldo/PrimacyRL/blob/main/dopamine_prl.ipynb" in order to Install Dopamine:

Code:
```ruby
  !git clone https://github.com//Mattia-Colbertaldo/dopamine_restart
```


# Third Step
In "https://github.com/Mattia-Colbertaldo/dopamine_restart/blob/master/dopamine/discrete_domains/run_experiment.py", add to the run_experiment definition:

Code:
```ruby
  def run_experiment(self):
    ...
      if iteration % 25 == 0:
        self._agent.ResetLastLayers()
    ...
```

In this way we reset the last layers every 25 iterations.

# Fourth Step

In "https://github.com/Mattia-Colbertaldo/dopamine_restart/blob/master/dopamine/agents/dqn/dqn_agent.py", define ResetLastLayers: it calls reset_last_layer for both online_convnet and target_convnet networks

Code:
```ruby
  def ResetLastLayers(self):
    self.online_convnet.reset_last_layer()
    self.target_convnet.reset_last_layer()
    self._net_outputs = self.online_convnet(self.state_ph)
```
