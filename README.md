# UVIP: Model-Free Approach to Evaluate Reinforcement Learning Algorithms

This is the implementation of model-free upper value iteration procedure (**UVIP**) that allows us to estimate the suboptimality gap $V^*(x) - V^{\pi}(x)$. 
The evaluation can be performed on the arbitrary policy $\pi$. The **UVIP** was tested on discrete state space environments (`Frozen Lake`, `Chain` and `Garnet`) and continuous ones (`CartPole` and `Acrobot`). 
