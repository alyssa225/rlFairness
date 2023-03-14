import numpy as np
import src.random


class QLearning:
    """
    QLearning reinforcement learning agent.

    Arguments:
      epsilon - (float) The probability of randomly exploring the action space
        rather than exploiting the best action.
      alpha - (float) The weighting to give current rewards in estimating Q. This 
        should range [0,1], where 0 means "don't change the Q estimate based on 
        current reward" 
      gamma - (float) This is the weight given to expected future rewards when 
        estimating Q(s,a). It should be in the range [0,1]. Here 0 means "don't
        incorporate estimates of future rewards into the reestimate of Q(s,a)"

      See page 131 of Sutton and Barto's Reinforcement Learning book for
        pseudocode and for definitions of alpha, gamma, epsilon 
        (http://incompleteideas.net/book/RLbook2020.pdf).  
    """

    def __init__(self, epsilon=0.2, alpha=0.5, gamma=0.5):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def fit(self, env, steps=1000, num_bins=100):
        """
        Trains an agent using Q-Learning on an OpenAI Gym Environment.

        See page 131 of Sutton and Barto's book Reinforcement Learning for
        pseudocode (http://incompleteideas.net/book/RLbook2020.pdf).
        Initialize your parameters as all zeros. Choose actions with
        an epsilon-greedy approach Note that unlike the pseudocode, we are
        looping over a total number of steps, and not a total number of
        episodes. This allows us to ensure that all of our trials have the same
        number of steps--and thus roughly the same amount of computation time.

        See (https://www.gymlibrary.dev/) for examples of how to use the OpenAI
        Gym Environment interface.

        In every step of the fit() function, you should sample
            two random numbers using functions from `src.random`.
            1.  First, use either `src.random.rand()` or `src.random.uniform()`
                to decide whether to explore or exploit.
            2. Then, use `src.random.choice` or `src.random.randint` to decide
                which action to choose. Even when exploiting, you should make a
                call to `src.random` to break (possible) ties.

        Please don't use `np.random` functions; use the ones from `src.random`!
        Please do not use `env.action_space.sample()`!

        Hints:
          - Use env.action_space.n and env.observation_space.n to get the
            number of available actions and states, respectively.
          - Remember to reset your environment at the end of each episode. To
            do this, call env.reset() whenever the value of "terminated or truncated" returned
            from env.step() is True.
          - In addition to resetting the environment, calling env.reset() will
            return the environment's initial state.
          - When choosing to exploit the best action rather than exploring,
            do not use np.argmax: it will deterministically break ties by
            choosing the lowest index of among the tied values. Instead,
            please *randomly choose* one of those tied-for-the-largest values.

        Arguments:
          env - (Env) An OpenAI Gym environment with discrete actions and
            observations. See the OpenAI Gym documentation for example use
            cases (https://www.gymlibrary.dev/api/core/).
          steps - (int) The number of actions to perform within the environment
            during training.

        Returns:
          state_action_values - (np.array) The values assigned by the algorithm
            to each state-action pair as a 2D numpy array. The dimensionality
            of the numpy array should be S x A, where S is the number of
            states in the environment and A is the number of possible actions.
          rewards - (np.array) A 1D sequence of averaged rewards of length num_bins.
            Let s = int(np.ceil(steps / num_bins)), then rewards[0] should
            contain the average reward over the first s steps, rewards[1]
            should contain the average reward over the next s steps, etc.
        """
        # set up rewards list, Q(s, a) table
        n_actions, n_states = env.action_space.n, env.observation_space.n
        state_action_values = np.zeros((n_states, n_actions))
        avg_rewards = np.zeros([num_bins])
        all_rewards = []
        # self.Q = np.zeros((n_states, n_actions))
        current_state, _ = env.reset()
        print('current_state: ', current_state)
        s = int(np.ceil(steps / num_bins))
        for i in range(steps):
          # current_state=0
          # truncated = False
          # while not truncated:
          p = src.random.rand()
          # print('prob: ', p)
          maxv = 0
          maxi=[]
          for i, savalue in enumerate(state_action_values[int(current_state),:]):
            if savalue>maxv:
              maxv = savalue
              maxi=[]
              maxi.append(i)
            elif savalue == maxv:
              maxi.append(i)
          print(maxi)
          if p < self.epsilon:
            # print('exploit')
            # print('Q: ', self.Q)
            for i, savalue in enumerate(state_action_values[int(current_state),:]):
              if savalue>maxv:
                maxv = savalue
                maxi=[]
                maxi.append(i)
              elif savalue == maxv:
                maxi.append(i)
              print(maxi)
            A=src.random.choice(maxi)         
          else:
            # print('explore')
            A=src.random.randint(n_actions)
          # print('A: ', A)
          # print('env.step: ',env.step(A))
          out = env.step(int(A))
          # print('env.step: ',out)
          # print('S_prime: ',out[0])
          # print('rewards: ',out[1])
          rewards = out[1]
          S_prime = out[0]
          all_rewards.append(out[1])
          # print("all_rewards: ", all_rewards)
          if i % s == 0 and i!=0:
            # print('i: ', i)
            # print('s: ', s)
            # print('i/s-1: ', i/s-1)
            avg_rewards[int(i/s-1)] = np.mean(all_rewards)
            all_rewards = []
          # print('na: ', n_actions)
          # print('Q: ', self.Q)
          # print("Q[s',na]: ", self.Q[int(S_prime),n_actions])
          
          maxv = 0
          maxi=[]
          for i, savalue in enumerate(state_action_values[int(S_prime),:]):
            if savalue>maxv:
              maxv = savalue
              maxi=[]
              maxi.append(i)
            elif savalue == maxv:
              maxi.append(i)
          print(maxi)
          a = src.random.choice(maxi)
          state_action_values[int(current_state),int(A)] = state_action_values[int(current_state),int(A)] + self.alpha*(rewards+self.gamma*state_action_values[int(S_prime),a]-state_action_values[int(current_state),int(A)])
          # state_action_values[int(out[0]),int(A)] = self.Q[int(current_state),int(A)]
          current_state = S_prime
          if out[2] == True or out[3]==True:
            current_state, _ = env.reset()
          # print('current_state: ', current_state)
          print('state_action_values: ', state_action_values)
        return state_action_values, avg_rewards


    def predict(self, env, state_action_values):
        """
        Runs prediction on an OpenAI environment using the policy defined by
        the QLearning algorithm and the state action values. Predictions are
        run for exactly one episode. Note that one episode may produce a
        variable number of steps.

        Hints:
          - You should not update the state_action_values during prediction.
          - Exploration is only used in training. During prediction, you
            should only "exploit."
          - In addition to resetting the environment, calling env.reset() will
            return the environment's initial state
          - You should use a loop to predict over each step in an episode until
            it terminates by returning `terminated or truncated=True`.
          - When choosing to exploit the best action, do not use np.argmax: it
            will deterministically break ties by choosing the lowest index of
            among the tied values. Instead, please *randomly choose* one of
            those tied-for-the-largest values.

        Arguments:
          env - (Env) An OpenAI Gym environment with discrete actions and
            observations. See the OpenAI Gym documentation for example use
            cases (https://www.gymlibrary.dev/).
          state_action_values - (np.array) The values assigned by the algorithm
            to each state-action pair as a 2D numpy array. The dimensionality
            of the numpy array should be S x A, where S is the number of
            states in the environment and A is the number of possible actions.

        Returns:
          states - (np.array) The sequence of states visited by the agent over
            the course of the episode. Does not include the starting state.
            Should be of length K, where K is the number of steps taken within
            the episode.
          actions - (np.array) The sequence of actions taken by the agent over
            the course of the episode. Should be of length K, where K is the
            number of steps taken within the episode.
          rewards - (np.array) The sequence of rewards received by the agent
            over the course  of the episode. Should be of length K, where K is
            the number of steps taken within the episode.
        """

        # setup
        n_actions, n_states = env.action_space.n, env.observation_space.n
        states, actions, rewards = [], [], []

        # reset environment before your first action
        current_state, _ = env.reset()
        truncated = False
        print('state_action_values: ', state_action_values)
        while not truncated:
          max = 0
          maxi=[]
          for i, savalue in enumerate(state_action_values[current_state,:]):
            if savalue>max:
              max = savalue
              maxi=[]
              maxi.append(i)
            elif savalue == max:
              maxi.append(i)
          # print('max ind: ', maxi)
          A=src.random.choice(maxi)
          out = env.step(int(A))
          if out[2] == True or out[3]==True:
            truncated = True 
          current_state = out[0]
          states.append(out[0])
          actions.append(A)
          rewards.append(out[1])
        np.array(states)
        np.array(actions)
        np.array(rewards)
        # raise NotImplementedError
        return states, actions, rewards
        # raise NotImplementedError
