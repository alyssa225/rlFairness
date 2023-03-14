import numpy as np
import src.random


class MultiArmedBandit:
    """
    MultiArmedBandit reinforcement learning agent.

    Arguments:
      epsilon - (float) The probability of randomly exploring the action space
        rather than exploiting the best action.
    """

    def __init__(self, epsilon=0.2):
        self.epsilon = epsilon

    def fit(self, env, steps=1000, num_bins=100):
        """
        Trains the MultiArmedBandit on an OpenAI Gym environment.

        See page 32 of Sutton and Barto's book Reinformcement Learning for
        pseudocode (http://incompleteideas.net/book/RLbook2020.pdf).
        Initialize your parameters as all zeros. For the step size, use
        1/N, where N is the number of times the current action has been
        performed. (This is the version of Bandits we saw in lecture before
        we introduced alpha). Use an epsilon-greedy approach to pick actions.

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
          - When choosing to exploit the best action rather than exploring,
            do not use np.argmax: it will deterministically break ties by
            choosing the lowest index of among the tied values. Instead,
            please *randomly choose* one of those tied-for-the-largest values.
          - MultiArmedBandit treats all environment states the same. However,
            in order to have the same API as agents that model state, you must
            explicitly return the state-action-values Q(s, a). To do so, just
            copy the action values learned by MultiArmedBandit S times, where
            S is the number of states.

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
          rewards - (np.array) A 1D sequence of averaged rewards of length `num_bins`.
            Let s = int(np.ceil(steps / `num_bins`)), then rewards[0] should
            contain the average reward over the first s steps, rewards[1]
            should contain the average reward over the next s steps, etc.
        """
        # set up Q function, rewards
        n_actions, n_states = env.action_space.n, env.observation_space.n
        self.Q = np.zeros(n_actions)
        self.N = np.zeros(n_actions)
        avg_rewards = np.zeros([num_bins])
        all_rewards = []
        state_action_values = np.zeros((n_states, n_actions))
        print('env.action_space: ', env.action_space)
        print('n_actions: ', n_actions)
        print('n_states: ', n_states)
        print('epsilon: ', self.epsilon)
        # reset environment before your first action
        env.reset()
        s = int(np.ceil(steps / num_bins))
        for i in range(steps):
          p = src.random.rand()
          # print('prob: ', p)
          if p < self.epsilon:
            # print('exploit')
            # print('Q: ', self.Q)
            A=src.random.choice(np.arange(self.Q.shape[0]))
          else:
            # print('explore')
            A=src.random.choice(n_actions)
          # print('A: ', A)
          # print('env.step: ',env.step(A))
          out = env.step(int(A))
          # print('env.step: ',out)
          # print('state: ',out[0])
          # print('rewards: ',out[1])
          rewards = out[1]
          all_rewards.append(out[1])
          # print("all_rewards: ", all_rewards)
          if i % s == 0 and i!=0:
            # print('i: ', i)
            # print('s: ', s)
            # print('i/s-1: ', i/s-1)
            avg_rewards[int(i/s-1)] = np.mean(all_rewards)
            all_rewards = []
          self.N[int(A)] += 1
          self.Q[int(A)] = self.Q[int(A)] + 1/self.N[int(A)]*(rewards-self.Q[int(A)])
          state_action_values[int(out[0]),int(A)] = self.Q[int(A)]
          # print('state_action_values: ', state_action_values)
        return state_action_values, avg_rewards

    def predict(self, env, state_action_values):
        """
        Runs prediction on an OpenAI environment using the policy defined by
        the MultiArmedBandit algorithm and the state action values. Predictions
        are run for exactly one episode. Note that one episode may produce a
        variable number of steps.

        Hints:
          - You should not update the state_action_values during prediction.
          - Exploration is only used in training. During prediction, you
            should only "exploit."
          - You should use a loop to predict over each step in an episode until
            it terminates by returning `terminated or truncated=True`.
          - When choosing to exploit the best action, do not use np.argmax: it
            will deterministically break ties by choosing the lowest index of
            among the tied values. Instead, please *randomly choose* one of
            those tied-for-the-largest values.

        Arguments:
          env - (Env) An OpenAI Gym environment with discrete actions and
            observations. See the OpenAI Gym documentation for example use
            cases (https://www.gymlibrary.dev/api/core/).
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
        # reset environment before your first action
        env.reset()
        states = []
        actions = []
        rewards = []
        truncated = False
        S = 0
        print('state_action_values: ', state_action_values)
        while not truncated:
          max = 0
          maxi=[]
          for i, savalue in enumerate(state_action_values[S,:]):
            if savalue>max:
              max = savalue
              maxi=[]
              maxi.append(i)
            elif savalue == max:
              maxi.append(i)
          print('max ind: ', maxi)
          A=src.random.choice(maxi)
          print('A: ',A)
          out = env.step(int(A))
          print('out: ',out)
          if out[2] == True or out[3]==True:
            truncated = True 
          S = out[0]
          states.append(out[0])
          actions.append(A)
          rewards.append(out[1])
        np.array(states)
        np.array(actions)
        np.array(rewards)
        # raise NotImplementedError
        return states, actions, rewards