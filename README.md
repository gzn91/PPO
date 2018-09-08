# Proximal Policy Optimization

    My my implementation of PPO based on OpenAI's baseline implementation.

## Layout:
    * Gym wrappers
        * GymWrap: Ordinary wrapper for continuous environments like RoboSchool
        * StackObs: Used for partially observable environments in order to estimate velocity acceleration etc..
        * ResetEarly: Can be used combined with GRU or LSTM models in partially observable environments,
         however this training process is very slow.
     
    * Models
        * MLP: Standard 2 hidden layers FC neural network, where the actor and critic network are independent.
        * CNN: Shared CNN-block followed by separate output layers for the actor and critic. 
        * LSTM and GRU: Recurrent neural network models that can model partially observable environments without
        the help of stacked frames. (Note that these should be used when the episode steps < the minibatch size)
     
    The models can be edited in ppo.py
    
## Main
    main_atari.py will run Brakeout by default.
    main_roboschool will run InvertedPendulum by default.
    
    The environments and other hyperparameters can be changed in the tensorflow flags.
    
![](gif/brakeout.gif)
