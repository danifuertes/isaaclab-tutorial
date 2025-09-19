import torch
from skrl.utils.spaces.torch import unflatten_tensorized_space
from skrl.models.torch import CategoricalMixin, DeterministicMixin, Model

class MLP(CategoricalMixin, DeterministicMixin, Model):
    
    def __init__(
        self,
        observation_space,
        action_space,
        device="cuda:0",
        hidden_dim=128,
        unnormalized_log_prob=True,
    ):
        Model.__init__(self, observation_space, action_space, device)
        CategoricalMixin.__init__(self, unnormalized_log_prob, role="policy")
        DeterministicMixin.__init__(self, False, role="value")
        self._shared_output = None
        
        # Number of dimensions
        self.num_dims = self.observation_space['agent'].shape[-1]
        self.obs_dims = self.observation_space['goal'].shape[-1]

        # Linear embeddings
        self.agent_linear = torch.nn.Sequential(
            torch.nn.Linear(self.num_dims, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU()
        )
        self.goal_linear = torch.nn.Sequential(
            torch.nn.Linear(self.obs_dims, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU()
        )

        # Actor (Policy)
        self.mean_layer = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, self.num_actions),
        )

        # Critic (Value)
        self.value_layer = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def act(self, inputs, role):
        # Override the act method to disambiguate its call
        if role == "policy":
            return CategoricalMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        
        # Inputs
        data = unflatten_tensorized_space(self.observation_space, inputs['states'])
        agent = data['agent']
        goal = data['goal']
        
        # Policy computation
        if role == "policy":
            
            # Embeddings
            agent_embed = self.agent_linear(agent)
            goal_embed = self.goal_linear(goal)
            self._shared_output = torch.cat([agent_embed, goal_embed], dim=-1)
            
            # Get mean and std
            mean = self.mean_layer(self._shared_output)
            return mean, {}
        
        # Value computation
        elif role == "value":
            
            # Embeddings
            if self._shared_output is None:
                agent_embed = self.agent_linear(agent)
                goal_embed = self.goal_linear(goal)
                shared_output = torch.cat([agent_embed, goal_embed], dim=-1)
            else:
                shared_output = self._shared_output.clone()
            
            # Reset saved shared output to prevent the use of erroneous data in subsequent steps
            self._shared_output = None
            
            # Get value
            value = self.value_layer(shared_output)
            return value, {}
