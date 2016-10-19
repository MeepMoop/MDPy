#!/usr/bin/env python

from random import random

class MDP:
  def __init__(self):
    self._mdp = []

  # add states to an MDP
  def add_states(self, num_states):
    for s in range(num_states):
      self._mdp.append([])

  # add actions to a state in an MDP
  def add_actions(self, state, num_actions):
    for a in range(num_actions):
      self._mdp[state].append([])

  # add a transition tuple (s', r, P) to a state-action pair in an MDP
  def add_transition(self, state, action, transition):
    self._mdp[state][action].append(transition)

  # removes a state from an MDP
  def remove_state(self, index):
    self._mdp.pop(index)

  # removes an action from a state in an MDP
  def remove_action(self, state, index):
    self._mdp[state].pop(index)

  # removes a transtion from a state-action pair in an MDP
  def remove_transition(self, state, action, index):
    self._mdp[state][action].pop(index)

  # clears all states from an MDP
  def clear_states(self):
    self._mdp = []

  # clears all actions from a state in an MDP
  def clear_actions(self, state):
    self._mdp[state] = []

  # clears all transitions from a state-action pair in an MDP
  def clear_transitions(self, state, action):
    self._mdp[state][action] = []

  # returns the number of states in an MDP
  def num_states(self):
    return len(self._mdp)

  # returns the number of actions in a state in an MDP
  def num_actions(self, state):
    return len(self._mdp[state])

  # returns the number of transitions in a state-action pair in an MDP
  def num_transitions(self, state, action):
    return len(self._mdp[state][action])

  # returns an array with the probabilities of each transition
  def get_probabilities(self, state, action):
    P = []
    for tr in range(len(self._mdp[state][action])):
      P.append(self._mdp[state][action][tr][2])
    return P

  # return a tuple with a next state and reward sampled from the transitions of a state-action pair
  def do_action(self, state, action):
    P = self.get_probabilities(state, action)
    sample = random()
    thresh = 0
    for tr in range(len(P)):
      thresh += P[tr]
      if sample < thresh:
        return (self._mdp[state][action][tr][0], self._mdp[state][action][tr][1])

  # returns an array containing the optimal value function for an MDP
  def value_iteration(self, gamma, tolerance=1e-6):
    v = [0.0] * self.num_states()
    dv = tolerance
    while dv >= tolerance:
      dv = 0.0
      vi = list(v)
      for s in range(self.num_states()):
        if self.num_actions(s) == 0:
          v[s] = 0.0
        else:
          ret_max = -1000000.0
          for a in range(self.num_actions(s)):
            ret = 0.0
            for tr in range(self.num_transitions(s, a)):
              tr_i = self._mdp[s][a][tr]
              ret += tr_i[2] * (tr_i[1] + gamma * v[tr_i[0]])
            if ret > ret_max:
              ret_max = ret
          v[s] = ret_max
        if abs(v[s] - vi[s]) > dv:
          dv = abs(v[s] - vi[s])
    return v

  # returns an array containing the optimal value function under a policy P[s][a] for an MDP
  def value_policy(self, P, gamma, tolerance=1e-6):
    v = [0.0] * self.num_states()
    dv = tolerance
    while dv >= tolerance:
      dv = 0.0
      vi = list(v)
      for s in range(self.num_states()):
        ret = 0.0
        for a in range(self.num_actions(s)):
          for tr in range(self.num_transitions(s, a)):
            tr_i = self._mdp[s][a][tr]
            ret += P[s][a] * tr_i[2] * (tr_i[1] + gamma * v[tr_i[0]])
        v[s] = ret
        if abs(v[s] - vi[s]) > dv:
          dv = abs(v[s] - vi[s])
    return v

  # returns an array containing the optimal value function under the equiprobable random policy for an MDP
  def value_equiprobable(self, gamma, tolerance=1e-6):
    v = [0.0] * self.num_states()
    dv = tolerance
    while dv >= tolerance:
      dv = 0.0
      vi = list(v)
      for s in range(self.num_states()):
        ret = 0.0
        for a in range(self.num_actions(s)):
          for tr in range(self.num_transitions(s, a)):
            tr_i = self._mdp[s][a][tr]
            ret += (1 / float(self.num_actions(s))) * tr_i[2] * (tr_i[1] + gamma * v[tr_i[0]])
        v[s] = ret
        if abs(v[s] - vi[s]) > dv:
          dv = abs(v[s] - vi[s])
    return v

  # returns an array Q[s][a] containing the optimal action-value function for an MDP
  def Q_iteration(self, gamma, tolerance=1e-6):
    return self.value_to_Q(self.value_iteration(gamma, tolerance), gamma)

  # returns an array Q[s][a] containing the optimal value function under a policy P[s][a] for an MDP
  def Q_policy(self, P, gamma, tolerance=1e-6):
    return self.value_to_Q(self.value_policy(P, gamma, tolerance), gamma)

  # returns an array Q[s][a] containing the optimal value function under the equiprobable random policy for an MDP
  def Q_equiprobable(self, gamma, tolerance=1e-6):
    return self.value_to_Q(self.value_equiprobable(gamma, tolerance), gamma)

  # returns an array Q[s][a] for an MDP given a value function and discount rate
  def value_to_Q(self, v, gamma):
    Q = []
    for s in range(self.num_states()):
      Q.append([0.0] * self.num_actions(s))
      for a in range(self.num_actions(s)):
        for tr in range(self.num_transitions(s, a)):
          tr_i = self._mdp[s][a][tr]
          Q[s][a] += tr_i[2] * (tr_i[1] + gamma * v[tr_i[0]])
    return Q

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
  print 'V[s]   ', mdp.value_iteration(0.9)
  print 'Q[s][a]', mdp.Q_iteration(0.9)

if __name__ == '__main__':
  example()
