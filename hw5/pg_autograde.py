import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm as _tqdm

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer

class NNPolicy(nn.Module):
    
    def __init__(self, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, num_hidden)
        self.l2 = nn.Linear(num_hidden, 2)

    def forward(self, x):
        """
        Performs a forward pass through the network.
        
        Args:
            x: input tensor (first dimension is a batch dimension)
            
        Return:
            Probabilities of performing all actions in given input states x. Shape: batch_size x action_space_size
        """
        # YOUR CODE HERE
        model = torch.nn.Sequential(
            self.l1,
            nn.ReLU(),
            self.l2,
            nn.Softmax(dim = 1)
        )
        return model(x.float())
        
    def get_probs(self, obs, actions):
        """
        This function takes a tensor of states and a tensor of actions and returns a tensor that contains 
        a probability of perfoming corresponding action in all states (one for every state action pair). 

        Args:
            obs: a tensor of states. Shape: batch_size x obs_dim
            actions: a tensor of actions. Shape: batch_size x 1

        Returns:
            A torch tensor filled with probabilities. Shape: batch_size x 1.
        """
        # YOUR CODE HERE
        probs = self.forward(obs)
        actions = actions.squeeze(1)
        
#         print('actions', actions, actions.shape)
#         print('probs', probs, probs.shape)

        # Selects the probability from the action
        # Should be rewritten with indexing instead of for loop
        action_probs = torch.tensor([prob[action] for prob, action in zip(probs, actions)])
        action_probs = action_probs.unsqueeze(1)
        
        return action_probs
    
    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            obs: state as a tensor. Shape: 1 x obs_dim or obs_dim

        Returns:
            An action (int).
        """
        # YOUR CODE HERE
        # Reshape if necessary, input shape is variable
        assert len(obs.shape) in [1, 2]
        if len(obs.shape) == 1: obs = obs.unsqueeze(0)
        
        probs = self.forward(obs).squeeze(0).detach().numpy() # Put observation through network
        action = np.random.choice([0, 1], p = probs) # Select action with weighted by network output
        
        return action

def sample_episode(env, policy):
    """
    A sampling routine. Given environment and a policy samples one episode and returns states, actions, rewards
    and dones from environment's step function as tensors.

    Args:
        env: OpenAI gym environment.
        policy: A policy which allows us to sample actions with its sample_action method.

    Returns:
        Tuple of tensors (states, actions, rewards, dones). All tensors should have same first dimension and 
        should have dim=2. This means that vectors of length N (states, rewards, actions) should be Nx1.
        Hint: Do not include the state after termination in states.
    """
    states = []
    actions = []
    rewards = []
    dones = []
    
    # YOUR CODE HERE
    state = env.reset()
    done = False
    
    while not done:
        action = policy.sample_action(torch.from_numpy(state))
        new_state, reward, done, info = env.step(action)
        
        states.append(torch.as_tensor(state))
        actions.append(torch.as_tensor(action))
        rewards.append(torch.as_tensor(reward))
        dones.append(torch.as_tensor(done))
        
        state = new_state
    
    
    states = torch.stack(states)
    actions = torch.stack(actions).unsqueeze(1)
    rewards = torch.stack(rewards).unsqueeze(1)
    dones = torch.stack(dones).unsqueeze(1)
    
    print(states.shape, actions.shape, rewards.shape, dones.shape)
    
    return states, actions, rewards, dones

def sample_episode(env, policy):
    """
    A sampling routine. Given environment and a policy samples one episode and returns states, actions, rewards
    and dones from environment's step function as tensors.

    Args:
        env: OpenAI gym environment.
        policy: A policy which allows us to sample actions with its sample_action method.

    Returns:
        Tuple of tensors (states, actions, rewards, dones). All tensors should have same first dimension and 
        should have dim=2. This means that vectors of length N (states, rewards, actions) should be Nx1.
        Hint: Do not include the state after termination in states.
    """
    states = []
    actions = []
    rewards = []
    dones = []
    
    # YOUR CODE HERE
    state = env.reset()
    done = False
    
    while not done:
        action = policy.sample_action(torch.from_numpy(state))
        new_state, reward, done, info = env.step(action)
        print(reward)
        states.append(torch.as_tensor(state))
        actions.append(torch.as_tensor(action))
        rewards.append(torch.as_tensor(reward))
        dones.append(torch.as_tensor(done))
        
        state = new_state
    
    
    states = torch.stack(states)
    actions = torch.stack(actions).unsqueeze(1)
    rewards = torch.stack(rewards).unsqueeze(1)
    dones = torch.stack(dones).unsqueeze(1)
    
    print(states.shape, actions.shape, rewards.shape, dones.shape)
    
    return states, actions, rewards, dones

def sample_episode(env, policy):
    """
    A sampling routine. Given environment and a policy samples one episode and returns states, actions, rewards
    and dones from environment's step function as tensors.

    Args:
        env: OpenAI gym environment.
        policy: A policy which allows us to sample actions with its sample_action method.

    Returns:
        Tuple of tensors (states, actions, rewards, dones). All tensors should have same first dimension and 
        should have dim=2. This means that vectors of length N (states, rewards, actions) should be Nx1.
        Hint: Do not include the state after termination in states.
    """
    states = []
    actions = []
    rewards = []
    dones = []
    
    # YOUR CODE HERE
    state = env.reset()
    done = False
    
    while not done:
        action = policy.sample_action(torch.from_numpy(state))
        new_state, reward, done, info = env.step(action)
        states.append(torch.as_tensor(state))
        actions.append(torch.as_tensor(action))
        rewards.append(torch.as_tensor(reward))
        dones.append(torch.as_tensor(done))
        
        state = new_state
    
    
    states = torch.stack(states)
    actions = torch.stack(actions).unsqueeze(1)
    rewards = torch.stack(rewards).unsqueeze(1)
    dones = torch.stack(dones).unsqueeze(1)
    
    print(states.shape, actions.shape, rewards.shape, dones.shape)
    
    return states, actions, rewards, dones

def compute_reinforce_loss(policy, episode, discount_factor):
    """
    Computes reinforce loss for given episode.

    Args:
        policy: A policy which allows us to get probabilities of actions in states with its get_probs method.

    Returns:
        loss: reinforce loss
    """
    # Compute the reinforce loss
    # Make sure that your function runs in LINEAR TIME
    # Note that the rewards/returns should be maximized 
    # while the loss should be minimized so you need a - somewhere
    
    # YOUR CODE HERE
    states, actions, rewards, dones = trajectory_data = episode
    action_probs = policy.get_probs(states, actions)
    
    states, actions, rewards, dones = trajectory_data
    action_probs = policy.get_probs(states, actions)
    loss = - torch.sum(torch.log(action_probs) * rewards, dim = 1)
    
    return loss

# YOUR CODE HERE
# raise NotImplementedError

def run_episodes_policy_gradient(policy, env, num_episodes, discount_factor, learn_rate, 
                                 sampling_function=sample_episode):
    optimizer = optim.Adam(policy.parameters(), learn_rate)
    
    episode_durations = []
    for i in range(num_episodes):
        
        # YOUR CODE HERE
        raise NotImplementedError
                           
        if i % 10 == 0:
            print("{2} Episode {0} finished after {1} steps"
                  .format(i, len(episode[0]), '\033[92m' if len(episode[0]) >= 195 else '\033[99m'))
        episode_durations.append(len(episode[0]))
        
    return episode_durations