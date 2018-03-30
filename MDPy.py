#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import scipy.optimize

class MDP(object):
  def __init__(self):
    self._mdp = []

  # add a number of states to the MDP
  def add_states(self, num_states):
    for s in range(num_states):
      self._mdp.append([])

  # add a number of actions to a state in the MDP
  def add_actions(self, state, num_actions):
    for a in range(num_actions):
      self._mdp[state].append([])

  # add a transition tuple (s', r, P) to a state-action pair in the MDP
  def add_transition(self, state, action, transition):
    self._mdp[state][action].append(transition)

  # remove a state from the MDP
  def remove_state(self, index):
    self._mdp.pop(index)

  # remove an action from a state in the MDP
  def remove_action(self, state, index):
    self._mdp[state].pop(index)

  # remove a transtion from a state-action pair in the MDP
  def remove_transition(self, state, action, index):
    self._mdp[state][action].pop(index)

  # clear all states from the MDP
  def clear_states(self):
    self._mdp = []

  # clear all actions from a state in the MDP
  def clear_actions(self, state):
    self._mdp[state] = []

  # clear all transitions from a state-action pair in the MDP
  def clear_transitions(self, state, action):
    self._mdp[state][action] = []

  # returns the number of states in the MDP
  def num_states(self):
    return len(self._mdp)

  # returns the number of actions in a state in the MDP
  def num_actions(self, state):
    return len(self._mdp[state])

  # returns the number of transitions in a state-action pair in the MDP
  def num_transitions(self, state, action):
    return len(self._mdp[state][action])

  # returns transition probabilities of a state-action pair in the MDP
  def get_probabilities(self, state, action):
    return [self._mdp[state][action][tr][2] for tr in range(self.num_transitions(state, action))]

  # return a tuple with a next state and reward sampled from the transitions of a state-action pair
  def do_action(self, state, action):
    tr = np.random.choice(self.num_transitions(state, action), p=self.get_probabilities(state, action))
    return self._mdp[state][action][tr][:-1]

  # returns state-values V[s] under a policy P(Q[s], s) for an MDP
  def value_policy(self, policy, gamma, tolerance=1e-6):
    V = [0.0 for s in range(self.num_states())]
    dv = tolerance
    while dv >= tolerance:
      dv = 0.0
      for s in range(self.num_states()):
        if self.num_actions(s) == 0:
          continue
        Vold = V[s]
        Q = np.zeros(self.num_actions(s))
        for a in range(self.num_actions(s)):
          ret = 0.0
          for tr in self._mdp[s][a]:
            ret += tr[2] * (tr[1] + gamma * V[tr[0]])
          Q[a] = ret
        V[s] = np.dot(policy(Q, s), Q)
        if abs(V[s] - Vold) > dv:
          dv = abs(V[s] - Vold)
    return V

  # returns state-values V[s] under a greedy policy for the MDP
  def value_iteration(self, gamma, tolerance=1e-6):
    pi = lambda Q, s: [1 if i == np.argmax(Q) else 0 for i in range(len(Q))]
    return self.value_policy(pi, gamma, tolerance)

  # returns state-values V[s] under an epsilon-greedy policy for the MDP
  def value_eps_greedy(self, epsilon, gamma, tolerance=1e-6):
    pi = lambda Q, s: [1 - epsilon + epsilon / len(Q) if i == np.argmax(Q) else epsilon / len(Q) for i in range(len(Q))]
    return self.value_policy(pi, gamma, tolerance)

  # returns state-values V[s] under an equiprobable random policy for the MDP
  def value_equiprobable(self, gamma, tolerance=1e-6):
    pi = lambda Q, s: [1 / len(Q) for i in range(len(Q))]
    return self.value_policy(pi, gamma, tolerance)

  # returns state-values V[s] under a tempered-softmax policy for the MDP
  def value_softmax(self, tau, gamma, tolerance=1e-6):
    pi = lambda Q, s: np.exp((Q - np.max(Q)) / tau) / np.exp((Q - np.max(Q)) / tau).sum()
    return self.value_policy(pi, gamma, tolerance)

  # returns state-values V[s] under a mellowmax policy for the MDP
  def value_mellowmax(self, omega, gamma, tolerance=1e-6, a=-1000, b=1000):
    def pi(Q, s):
      mm = np.max(Q) + np.log(np.exp(omega * (Q - np.max(Q))).mean()) / omega
      beta = scipy.optimize.brentq(lambda beta: np.sum(np.exp(beta * (Q - mm) - np.max(beta * (Q - mm))) * (Q - mm)), a=a, b=b)
      return np.exp(beta * (Q - np.max(Q))) / np.exp(beta * (Q - np.max(Q))).sum()
    return self.value_policy(pi, gamma, tolerance)

  # returns action-values Q[s][a] under a policy P(Q[s], s) for an MDP
  def Q_policy(self, policy, gamma, tolerance=1e-6):
    Q = [[0.0 for a in range(self.num_actions(s))] for s in range(self.num_states())]
    dv = tolerance
    while dv >= tolerance:
      dv = 0.0
      for s in range(self.num_states()):
        for a in range(self.num_actions(s)):
          Qold = Q[s][a]
          ret = 0.0
          for tr in self._mdp[s][a]:
            ret += tr[2] * (tr[1] + gamma * np.dot(policy(Q[tr[0]], tr[0]), Q[tr[0]]))
          Q[s][a] = ret
          if abs(Q[s][a] - Qold) > dv:
            dv = abs(Q[s][a] - Qold)
    return Q

  # returns action-values Q[s][a] under a greedy policy for the MDP
  def Q_iteration(self, gamma, tolerance=1e-6):
    pi = lambda Q, s: [1 if i == np.argmax(Q) else 0 for i in range(len(Q))]
    return self.Q_policy(pi, gamma, tolerance)

  # returns action-values Q[s][a] under an epsilon-greedy policy for the MDP
  def Q_eps_greedy(self, epsilon, gamma, tolerance=1e-6):
    pi = lambda Q, s: [1 - epsilon + epsilon / len(Q) if i == np.argmax(Q) else epsilon / len(Q) for i in range(len(Q))]
    return self.Q_policy(pi, gamma, tolerance)

  # returns action-values Q[s][a] under an equiprobable random policy for the MDP
  def Q_equiprobable(self, gamma, tolerance=1e-6):
    pi = lambda Q, s: [1 / len(Q) for i in range(len(Q))]
    return self.Q_policy(pi, gamma, tolerance)

  # returns action-values Q[s][a] under a tempered-softmax policy for the MDP
  def Q_softmax(self, tau, gamma, tolerance=1e-6):
    pi = lambda Q, s: np.exp((Q - np.max(Q)) / tau) / np.exp((Q - np.max(Q)) / tau).sum()#[np.exp((Q[i] - np.max(Q)) / tau) / np.sum(np.exp((Q - np.max(Q)) / tau)) for i in range(len(Q))]
    return self.Q_policy(pi, gamma, tolerance)

  # returns action-values Q[s][a] under a mellowmax policy for the MDP
  def Q_mellowmax(self, omega, gamma, tolerance=1e-6, a=-1000, b=1000):
    def pi(Q, s):
      mm = np.max(Q) + np.log(np.exp(omega * (Q - np.max(Q))).mean()) / omega
      beta = scipy.optimize.brentq(lambda beta: np.sum(np.exp(beta * (Q - mm) - np.max(beta * (Q - mm))) * (Q - mm)), a=a, b=b)
      return np.exp(beta * (Q - np.max(Q))) / np.exp(beta * (Q - np.max(Q))).sum()
    return self.Q_policy(pi, gamma, tolerance)

  # checks if a state is terminal
  def is_terminal(self, state):
    return self.num_actions(state) == 0

def example():
  # create an MDP
  mdp = MDP()

  # add 2 states
  mdp.add_states(2)

  # add 2 actions to the first state and 1 action to the second state
  mdp.add_actions(0, 2)
  mdp.add_actions(1, 1)

  # add transitions (s', r, P) for each state-action pair
  mdp.add_transition(0, 0, (0, 0.5, 1.0))
  mdp.add_transition(0, 1, (0, -1.0, 0.3))
  mdp.add_transition(0, 1, (1, -1.0, 0.7))
  mdp.add_transition(1, 0, (0, 5.0, 0.6))
  mdp.add_transition(1, 0, (1, -1.0, 0.4))

  # output optimal state-value and action-value functions with discount rate 0.9
  print('V[s]   ', mdp.value_iteration(0.9))
  print('Q[s][a]', mdp.Q_iteration(0.9))
  print('Q[s][a]', mdp.Q_equiprobable(0.9))

if __name__ == '__main__':
  example()
