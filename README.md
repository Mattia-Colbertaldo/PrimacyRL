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
# Forward Last Layer Restart
First, we add the restart to the last layer only. Below, we explain the steps forward in the Last Layer Restart process.
# First Step: Forward Last Layer Restart
Since in the Colab Notebook we use:
```ruby
DQNAgent.network = @gym_lib.CartpoleDQNNetwork
```
In "https://github.com/Mattia-Colbertaldo/dopamine_restart/blob/master/dopamine/agents/dqn/dqn_agent.py#L439" we added to the method __init__:
```ruby
  self.online_convnet_state = self.online_convnet.get_weights()
```
        
# Second Step: Forward Last Layer Restart    
Change the Colab Notebook so that clone our forked repo "https://github.com/Mattia-Colbertaldo/PrimacyRL/blob/main/dopamine_prl.ipynb" in order to Install Dopamine:

Code:
```ruby
  !git clone https://github.com//Mattia-Colbertaldo/dopamine_restart
```


# Third Step: Forward Last Layer Restart
In "https://github.com/Mattia-Colbertaldo/dopamine_restart/blob/master/dopamine/agents/dqn/dqn_agent.py", add to the ResetWeights method:

```ruby
  def ResetWeights(self):

    print("Resetting weights...")
    if self.reset_last_layer:
      print("Resetting last layer!")
      self.online_convnet.layers[-1].last_layer.kernel.initializer.run(session=self._sess)
      self.online_convnet.layers[-1].last_layer.bias.initializer.run(session=self._sess)

    if self.reset_dense1:
      print("Resetting dense1 layer!")
      self.online_convnet.layers[-1].dense1.kernel.initializer.run(session=self._sess)
      self.online_convnet.layers[-1].dense1.bias.initializer.run(session=self._sess)

    if self.reset_dense2:
      print("Resetting dense2 layer!")
      self.online_convnet.layers[-1].dense2.kernel.initializer.run(session=self._sess)
      self.online_convnet.layers[-1].dense2.bias.initializer.run(session=self._sess)

    # Reset the optimizer state
    optimizer_reset = tf.compat.v1.variables_initializer(self.optimizer_state)
    self._sess.run(optimizer_reset)
```

# Results
 Our primary objective was to substantiate the hypothesis that incorporating
the periodic resetting strategy enhances the overall score. 

![Alt text]((https://github.com/Mattia-Colbertaldo/PrimacyRL/blob/main/plot_2_both.png))
This plot illustrates the outcomes when resetting all three layers
at a frequency of 500 iterations with 100 evaluation steps. Notably, the mean scores showed a discernible improvement,
supporting the efficacy of the resetting approach. 


![Alt text](https://github.com/Mattia-Colbertaldo/PrimacyRL/blob/main/plot_3_both.png)
This plot presents results from resetting only the second
and last layers, conducted every 200 iterations with 1000 evaluation steps. This configuration yielded very promising
outcomes, further reinforcing the positive impact of periodic resetting on the DQN agent’s performance in the CartPole
environment.

Adding also the visualization of the variance we have:
![Alt text](https://github.com/Mattia-Colbertaldo/PrimacyRL/blob/main/mixed_plot_01.png.png)
![Alt text](https://github.com/Mattia-Colbertaldo/PrimacyRL/blob/main/mixed_plot_3.png.png)

Legend meaning:
DQN = no reset
DQN + reset = reset all the layers
DQN + reset 2 = reset only the last 2 the layers
