import json
import os
import datetime

def save_experiment_results(metadata, results, folder="results"):
    """
    Saves metadata and results data in a JSON file
    """
    os.makedirs(folder, exist_ok=True)
    
    # Generate filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = metadata.get("experiment_name", "exp")
    filename = f"{exp_name}_{timestamp}.json"
    filepath = os.path.join(folder, filename)
    
    data_to_save = {
        "metadata": metadata,
        "results": results
    }
    
    with open(filepath, 'w') as f:
        json.dump(data_to_save, f, indent=4)
    
    print(f"Experiment results saved in: {filepath}")
    return filepath