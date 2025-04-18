utils.py for weight and bias initialization, use only the def init() function

Action sampler: AT first, custom orthogonal weight and zero bias initialization is done, then action is sampled from a multi-agent env. action_num = number of agents. In the environment, there could be multiple agents, each of which would have its own distribution and pick an action from that distribution.
