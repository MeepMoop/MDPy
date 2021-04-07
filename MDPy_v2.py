import torch
import numpy as np

class MDP():
  def __init__(self, n_states, n_actions):
    self._mdp = [[[[], [], []] for a in range(n_actions)] for s in range(n_states)]

  # add a transition tuple (P, s', r) to a state-action pair in the MDP
  def add_transition(self, state, action, probability, next_state, reward):
    self._mdp[state][action][0].append(probability)
    self._mdp[state][action][1].append(next_state)
    self._mdp[state][action][2].append(reward)

  # returns the number of states in the MDP
  @property
  def n_states(self):
    return len(self._mdp)

  # returns the number of actions in a state in the MDP
  @property
  def n_actions(self):
    return len(self._mdp[0])

  # returns the number of transitions in a state-action pair in the MDP
  def n_transitions(self, state, action):
    return len(self._mdp[state][action][0])

  # checks if state is terminal (no transitions)
  def is_terminal(self, state):
    return len(self._mdp[state][0][0]) == 0

  # sample an outcome (s', r, T) from a state-action pair's possible transitions
  def sample(self, state, action):
    tr = np.random.choice(self.n_transitions(state, action), p=self._mdp[state][action][0])
    sp, r = self._mdp[state][action][1][tr], self._mdp[state][action][2][tr]
    return sp, r, self.is_terminal(sp)

  # compute vectorized form of MDP (needed for functions below)
  def build(self):
    self._P = np.zeros((self.n_states, self.n_states, self.n_actions))
    self._r = np.zeros((self.n_states, self.n_actions))
    for s in range(self.n_states):
      for a in range(self.n_actions):
        for tr in range(self.n_transitions(s, a)):
          self._P[self._mdp[s][a][1][tr], s, a] += self._mdp[s][a][0][tr]
          self._r[s, a] += self._mdp[s][a][0][tr] * self._mdp[s][a][2][tr]

  # get state to state transition matrix induced by a policy pi[s, a]
  def state_P_pi(self, pi):
    return (self._P * pi).sum(2).T

  # get expected immediate reward per state induced by a policy pi[s, a]
  def state_r_pi(self, pi):
    return (self._r * pi).sum(1, keepdims=True)

  # one sweep of dynamic programming for v under fixed policy pi[s, a]
  def v_dp(self, v, pi, gamma):
    return self.state_r_pi(pi) + gamma * self.state_P_pi(pi) @ v

  # compute state-values under fixed policy pi[s, a]
  def get_v_fixed_pi(self, pi, gamma):
    return np.linalg.inv(np.eye(self.n_states) - gamma * self.state_P_pi(pi)) @ self.state_r_pi(pi)

  # compute state-values under a policy derived from Q pi(q)[s, a] 
  def get_v_pi(self, pi, gamma):
    q = self.get_q_pi(pi, gamma)
    return (pi(q) * q).sum(1, keepdims=True)

  # compute state-values under equiprobable random policy
  def get_v_equiprobable(self, gamma):
    pi = np.ones((self.n_states, self.n_actions)) / self.n_actions
    return self.get_v_fixed_pi(pi, gamma)

  # compute state-values under epsilon-greedy policy
  def get_v_eps_greedy(self, epsilon, gamma):
    pi = lambda q: epsilon * np.ones(q.shape) / q.shape[1] + (1 - epsilon) * np.eye(q.shape[1])[q.argmax(1)]
    return self.get_v_pi(pi, gamma)

  # compute state-values under boltzmann policy
  def get_v_boltzmann(self, temperature, gamma):
    pi = lambda q: np.exp(temperature * (q - np.tile(q.max(1, keepdims=True), q.shape[1]))) / \
      np.tile(np.exp(temperature * (q - np.tile(q.max(1, keepdims=True), q.shape[1]))).sum(1, keepdims=True), q.shape[1])
    return self.get_v_pi(pi, gamma)

  # compute optimal state-values
  def get_v_star(self, gamma):
    pi = lambda q: np.eye(q.shape[1])[q.argmax(1)]
    return self.get_v_pi(pi, gamma)

  # get state-action to state-action transition matrix induced by a policy pi[s, a]
  def action_P_pi(self, pi):
    P_pi = np.reshape(self._P, (self.n_states, self.n_states * self.n_actions)).T
    P_pi = np.repeat(P_pi, self.n_actions).reshape(self.n_states * self.n_actions, self.n_states, self.n_actions) * pi
    return np.reshape(P_pi, (self.n_states * self.n_actions, self.n_states * self.n_actions))

  # get expected immediate reward per state-action pair
  def action_r_pi(self, pi=None):
    return np.reshape(self._r, (self.n_states * self.n_actions, 1))

  # one sweep of dynamic programming for q under fixed policy pi[s, a]
  def q_dp(self, q, pi, gamma):
    return (self.action_r_pi() + gamma * self.action_P_pi(pi) @ q.reshape((self.n_states * self.n_actions, 1))).reshape(self.n_states, self.n_actions)

  # compute action-values under fixed policy pi[s, a]
  def get_q_fixed_pi(self, pi, gamma):
    return (np.linalg.inv(np.eye(self.n_states * self.n_actions) - gamma * self.action_P_pi(pi)) @ self.action_r_pi(pi)).reshape((self.n_states, self.n_actions))

  # compute action-values under a policy derived from Q pi(q)[s, a] 
  def get_q_pi(self, pi, gamma):
    q_prev = np.zeros((self.n_states, self.n_actions))
    q = self.get_q_fixed_pi(pi(q_prev), gamma)
    while not np.allclose(q, q_prev):
      q_prev = q
      q = self.get_q_fixed_pi(pi(q_prev), gamma)
    return q

  # compute state-values under equiprobable random policy
  def get_q_equiprobable(self, gamma):
    pi = np.ones((self.n_states, self.n_actions)) / self.n_actions
    return self.get_q_fixed_pi(pi, gamma)

  # compute action-values under epsilon-greedy policy
  def get_q_eps_greedy(self, epsilon, gamma):
    pi = lambda q: epsilon * np.ones(q.shape) / q.shape[1] + (1 - epsilon) * np.eye(q.shape[1])[q.argmax(1)]
    return self.get_q_pi(pi, gamma)

  # compute action-values under boltzmann policy
  def get_q_boltzmann(self, temperature, gamma):
    pi = lambda q: np.exp(temperature * (q - np.tile(q.max(1, keepdims=True), q.shape[1]))) / \
      np.tile(np.exp(temperature * (q - np.tile(q.max(1, keepdims=True), q.shape[1]))).sum(1, keepdims=True), q.shape[1])
    return self.get_q_pi(pi, gamma)

  # compute optimal action-values
  def get_q_star(self, gamma):
    pi = lambda q: np.eye(q.shape[1])[q.argmax(1)]
    return self.get_q_pi(pi, gamma)
