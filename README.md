# MDPy

MDPy is an MDP library for Python. It allows for the creation of arbitrary MDPs, simulating actions in then, and solving them through implementations of policy evaluation and value iteration (for both state-values and action-values).

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

# output optimal value and action-value functions
print 'V[s]   ', mdp.value_iteration(0.9, 1e-6)
print 'Q[s][a]', mdp.Q_iteration(0.9, 1e-6)
```
