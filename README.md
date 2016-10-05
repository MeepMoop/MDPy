# MDPy

MDPy is a simple MDP library for Python. It allows for the creation of arbitrary MDPs, simulating actions in then, and solving them through implementations of policy evaluation and value iteration (for both state-values and action-values).

# Usage

```python
from MDPy import MDP

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
print 'V[s]   ', mdp.value_iteration(0.9, 1e-6)
print 'Q[s][a]', mdp.Q_iteration(0.9, 1e-6)
```
```
V[s]    [7.858262284624602, 10.692908326840795]
Q[s][a] [[7.572436056162142, 7.858263062758344], [10.692908631359973]]
```
