import torch

def replace_values(pairs):
    """
    Replace values in multiple target state dictionaries with values from corresponding source state dictionaries.

    :param pairs: List of tuples, where each tuple contains (source_dict, target_dict)
    :return: List of modified target dictionaries
    """
    modified_targets = []
    for source, target in pairs:
        for key in target.keys():
            if key in source:
                print(f'{key} replaced')
                target[key] = source[key]
            else:
                raise KeyError(f"Key '{key}' not found in the target state dictionary.")
        modified_targets.append(target)
    return modified_targets

# Example file paths for the model weights
vidllava_tuned_weights_1 = r"cache_dir\models--LanguageBind--Video-LLaVA-7B\pytorch_model-00001-of-00002.bin"
vidllava_tuned_weights_2 = r"cache_dir\models--LanguageBind--Video-LLaVA-7B\pytorch_model-00002-of-00002.bin"

to_replace_vicuna_weights1 = r"cache_dir\lmsys\vicuna-vidllava-tuned-7b-v1.5\pytorch_model-00001-of-00002.bin"
to_replace_vicuna_weights2 = r"cache_dir\lmsys\vicuna-vidllava-tuned-7b-v1.5\pytorch_model-00002-of-00002.bin"

# Load the model weights
source1 = torch.load(vidllava_tuned_weights_1, map_location=torch.device('cpu'))
target1 = torch.load(to_replace_vicuna_weights1, map_location=torch.device('cpu'))

source2 = torch.load(vidllava_tuned_weights_2, map_location=torch.device('cpu'))
target2 = torch.load(to_replace_vicuna_weights2, map_location=torch.device('cpu'))

# Prepare pairs of source and target dictionaries
replacement_pairs = [(source1, target1), (source2, target2)]

# Perform replacements and get the modified targets
modified_targets = replace_values(replacement_pairs)

# Save the modified weights
torch.save(modified_targets[0], to_replace_vicuna_weights1)
torch.save(modified_targets[1], to_replace_vicuna_weights2)

def print_dict_shapes(weights_dict):
    for key, value in weights_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"Key: {key}, Value Shape: {value.shape}")
        elif isinstance(value, dict):
            print(f"Key: {key}, Value is another dictionary:")
            print_dict_shapes(value)
        else:
            print(f"Key: {key}, Value is of type {type(value)}")

def save_tensor_to_text(file_path, tensor):
    with open(file_path, 'w') as file:
        for value in tensor.view(-1):
            file.write(f'{value.item()}\n')