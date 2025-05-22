import yaml
import os
import ast # For safely evaluating string representations of policy_kwargs

def load_config(config_path="configs/ppo_treasury_config.yaml"):
    """
    Loads a YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: A dictionary containing the configuration parameters.
              Returns None if the file cannot be loaded.
    """
    if not os.path.exists(config_path):
        print(f"ERROR: Configuration file not found at {config_path}")
        # In a real application, you might raise an error or use a default config
        return None 

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Example of post-processing: Convert policy_kwargs string to dict if needed
        # This is one way to handle complex structures in YAML if they are stored as strings.
        # A better way is to ensure the YAML loader can directly parse them (e.g. using !!python/object tags or just proper YAML structure)
        if isinstance(config.get('agent', {}).get('policy_kwargs'), str):
            try:
                config['agent']['policy_kwargs'] = ast.literal_eval(config['agent']['policy_kwargs'])
            except (ValueError, SyntaxError) as e:
                print(f"Warning: Could not evaluate agent.policy_kwargs string: {config['agent']['policy_kwargs']}. Error: {e}. Keeping as string.")

        print(f"Configuration loaded successfully from {config_path}")
        return config
        
    except yaml.YAMLError as e:
        print(f"ERROR: Error parsing YAML configuration file {config_path}: {e}")
        return None
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while loading config {config_path}: {e}")
        return None

if __name__ == '__main__':
    # Example Usage:
    # Assuming this script is in src/utils/ and configs/ is at project root.
    # Adjust path for standalone execution if necessary.
    
    # Construct path relative to this script's location to find the configs dir
    # This script is in .../src/utils/config_loader.py
    # Configs are in .../configs/
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_script_dir)) # up two levels from src/utils to project root
    example_config_path = os.path.join(project_root, "configs/ppo_treasury_config.yaml")

    print(f"Attempting to load example config from: {example_config_path}")
    
    if not os.path.exists(example_config_path):
        print(f"Example config file not found at {example_config_path}. Make sure it exists.")
        # Create a dummy one for testing if it doesn't exist
        print("Creating a dummy ppo_treasury_config.yaml in configs/ for testing config_loader.py.")
        os.makedirs(os.path.join(project_root, "configs"), exist_ok=True)
        dummy_cfg_content = {
            "project_name": "DummyProject",
            "data": {"raw_data_path": "dummy_path.csv"},
            "agent": {"policy": "MlpPolicy", "learning_rate": 0.001, "policy_kwargs": "{'net_arch': [{'pi': [32], 'vf': [32]}]}"} # String for policy_kwargs
        }
        with open(example_config_path, 'w') as f:
            yaml.dump(dummy_cfg_content, f)
            
    config_data = load_config(example_config_path)

    if config_data:
        print("\nSuccessfully loaded config data:")
        print(f"Project Name: {config_data.get('project_name')}")
        print(f"Raw data path: {config_data.get('data', {}).get('raw_data_path')}")
        print(f"Agent Policy: {config_data.get('agent', {}).get('policy')}")
        print(f"Agent LR: {config_data.get('agent', {}).get('learning_rate')}")
        
        policy_kwargs = config_data.get('agent', {}).get('policy_kwargs')
        print(f"Agent Policy Kwargs: {policy_kwargs}")
        if isinstance(policy_kwargs, dict):
            print("Policy kwargs successfully parsed as dict.")
        else:
            print("Policy kwargs is still a string or not found.")

        # Clean up dummy file if it was created by this test
        if 'dummy_cfg_content' in locals() and os.path.exists(example_config_path):
             # Check if it's the dummy one before removing, be careful
            if config_data.get("project_name") == "DummyProject":
                 print(f"Removing dummy config file: {example_config_path}")
                 os.remove(example_config_path)
    else:
        print("\nFailed to load config data.")
