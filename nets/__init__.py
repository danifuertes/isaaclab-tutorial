from .mlp import MLP

MODELS = {
    "mlp": MLP,
}

def list_models():
    """
    List all available models.
    """
    return list(MODELS.keys())

def load_model(model_name, env):
    
    # Check if model name is valid
    if model_name not in MODELS:
        return None
    
    # Get model with separated layers for policy (actor) and value (critic)
    if isinstance(MODELS[model_name], dict):
        policy = MODELS[model_name]["policy"]
        value = MODELS[model_name]["value"]
    
        # Load models
        models = {
            "policy": policy(env.observation_space, env.action_space, env.device),
            "value": value(env.observation_space, env.action_space, env.device)
        }
        
    # Get model with shared layers for policy (actor) and value (critic)
    else:
        policy = MODELS[model_name](env.observation_space, env.action_space, env.device)
        
        # Load models
        models = {
            "policy": policy,
            "value": policy
        }
    
    return models
