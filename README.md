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
In https://github.com/Mattia-Colbertaldo/dopamine_restart/blob/master/dopamine/discrete_domains/gym_lib.py, we added "reset_last_layer" method which resets the last fully connected layer by simply creating it.



and in "https://github.com/Mattia-Colbertaldo/dopamine_restart/blob/master/dopamine/agents/dqn/dqn_agent.py#L439" we added to the method __init__:
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
In "https://github.com/Mattia-Colbertaldo/dopamine_restart/blob/master/dopamine/agents/dqn/dqn_agent.py", add to the _train_step definition:

Code:
```ruby
 if self.training_steps % self.reset_period == 0:
   print("Resetting last layers...")
   self.ResetWeights()

  ...

