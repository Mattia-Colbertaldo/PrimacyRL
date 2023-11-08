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

# Enhancing Dopamine with Restart Functionality: A Step-by-Step Guide

# First Step
In dopamine_restart/dopamine/discrete_domains/atari_lib.py, we added "reset_last_layer" method which resets the last fully connected layer by simply creating it.

Code:
  def reset_last_layer(self):
        """Reset the last layer of the network."""
        self.dense2 = tf.keras.layers.Dense(self.num_actions, name='fully_connected')
        
# Second Step     
Change the colab so that clone our forked repo "https://github.com/Mattia-Colbertaldo/PrimacyRL/blob/main/dopamine_prl.ipynb" in order to Install Dopamine

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

# Forth Step

In "https://github.com/Mattia-Colbertaldo/dopamine_restart/blob/master/dopamine/agents/dqn/dqn_agent.py", define ResetLastLayers: it calls reset_last_layer for both online_convnet and target_convnet networks

Code:
```ruby
  def ResetLastLayers(self):
    self.online_convnet.reset_last_layer()
    self.target_convnet.reset_last_layer()
    self._net_outputs = self.online_convnet(self.state_ph)
```
