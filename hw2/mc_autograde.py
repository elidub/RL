import numpy as np
from collections import defaultdict
from tqdm import tqdm as _tqdm

def tqdm(*args, **kwargs):
    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer

class SimpleBlackjackPolicy(object):
    """
    A simple BlackJack policy that sticks with 20 or 21 points and hits otherwise.
    """
    def get_probs(self, states, actions):
        """
        This method takes a list of states and a list of actions and returns a numpy array that contains a probability
        of perfoming action in given state for every corresponding state action pair. 

        Args:
            states: a list of states.
            actions: a list of actions.

        Returns:
            Numpy array filled with probabilities (same length as states and actions)
        """
        # YOUR CODE HERE
        
        probs = [None] * len(actions)

        for i, (state, action) in enumerate(zip(states, actions)):
            
            sampled_action = self.sample_action(state)
            
            probs[i] = (sampled_action == action)*1.
            
        return np.array(probs)
    
    def sample_action(self, state):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            state: current state

        Returns:
            An action (int).
        """
        # YOUR CODE HERE
        
        stick, hit = 0, 1
        player, dealer, usable_ace = state

        if usable_ace and player > 21:
            player =- 10 
            
        action = hit if player < 20 else stick
            
        return action

def sample_episode(env, policy):
    """
    A sampling routine. Given environment and a policy samples one episode and returns states, actions, rewards
    and dones from environment's step function and policy's sample_action function as lists.

    Args:
        env: OpenAI gym environment.
        policy: A policy which allows us to sample actions with its sample_action method.

    Returns:
        Tuple of lists (states, actions, rewards, dones). All lists should have same length. 
        Hint: Do not include the state after the termination in the list of states.
    """
    states = []
    actions = []
    rewards = []
    dones = []
    
    # YOUR CODE HERE
    state = env.reset()
    done = False
    
    while not done:
        action = policy.sample_action(state)
        new_state, reward, done, info = env.step(action)
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        
        state = new_state
        
    return states, actions, rewards, dones

def mc_prediction(env, policy, num_episodes, discount_factor=1.0, sampling_function=sample_episode):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given policy using sampling.
    
    Args:
        env: OpenAI gym environment.
        policy: A policy which allows us to sample actions with its sample_action method.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(float)
    returns_count = defaultdict(float)
    
    # YOUR CODE HERE
    for _ in tqdm(range(num_episodes)):
        states, actions, rewards, dones = episode = sample_episode(env, policy)
        G = 0.
#         for state, action, reward, done in zip(*episode):
        reversed_episodes = zip(reversed(states), reversed(actions), reversed(rewards), reversed(dones))
        for t, (state, action, reward, done) in enumerate(reversed_episodes):
            G = discount_factor * G + reward
            if state not in states[:t]:
#                 print(returns_count, returns_count[state])
                returns_count[state] += 1
                V[state] = (returns_count[state] * V[state] + G) / (returns_count[state] + 1)
    
    return V

class RandomBlackjackPolicy(object):
    """
    A random BlackJack policy.
    """
    def get_probs(self, states, actions):
        """
        This method takes a list of states and a list of actions and returns a numpy array that contains 
        a probability of perfoming action in given state for every corresponding state action pair. 

        Args:
            states: a list of states.
            actions: a list of actions.

        Returns:
            Numpy array filled with probabilities (same length as states and actions)
        """
        # YOUR CODE HERE
        
        probs = [0.5] * len(actions)

#         for i, (state, action) in enumerate(zip(states, actions)):
            
#             sampled_action = self.sample_action(state)
            
#             probs[i] = (sampled_action == action)*1.
            
        return np.array(probs)
    
    def sample_action(self, state):
        """
        This method takes a state as input and returns an action sampled from this policy.  

        Args:
            state: current state

        Returns:
            An action (int).
        """
        # YOUR CODE HERE
        
        stick, hit = 0, 1
        action = np.random.choice([stick, hit])
        
        return action

def mc_importance_sampling(env, behavior_policy, target_policy, num_episodes, discount_factor=1.0,
                           sampling_function=sample_episode):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given target policy using behavior policy and ordinary importance sampling.
    
    Args:
        env: OpenAI gym environment.
        behavior_policy: A policy used to collect the data.
        target_policy: A policy which value function we want to estimate.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        sampling_function: Function that generates data from one episode.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

    # Keeps track of current V and count of returns for each state
    # to calculate an update.
    V = defaultdict(float)
    returns_count = defaultdict(float)
    
    # YOUR CODE HERE
    for _ in tqdm(range(num_episodes)):
#         behavior_policy = target_policy
        states, actions, rewards, dones = episode = sample_episode(env, behavior_policy)
        G = 0.
        W = 1.
        reversed_episodes = zip(reversed(states), reversed(actions), reversed(rewards), reversed(dones))
        for t, (state, action, reward, done) in enumerate(reversed_episodes):
            G = discount_factor * G + reward
            pi, b = target_policy.get_probs([state], [action]), behavior_policy.get_probs([state], [action])
            
            W = W * pi / b # Already calculate W_{n+1}
            V[state] = V[state] + 1 / (returns_count[state] + 1) * (W * G - V[state])

            if W == 0:
                break
    
    return V
