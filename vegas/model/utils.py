from transformers import AutoConfig


def auto_upgrade(config):
    cfg = AutoConfig.from_pretrained(config)
    if 'llava' in config and 'llava' not in cfg.model_type:
        assert cfg.model_type == 'llama'
        print("You are using newer LLaVA code base, while the checkpoint of v0 is from older code base.")
        print("You must upgrade the checkpoint to the new code base (this can be done automatically).")
        confirm = input("Please confirm that you want to upgrade the checkpoint. [Y/N]")
        if confirm.lower() in ["y", "yes"]:
            print("Upgrading checkpoint...")
            assert len(cfg.architectures) == 1
            setattr(cfg.__class__, "model_type", "llava")
            cfg.architectures[0] = 'LlavaLlamaForCausalLM'
            cfg.save_pretrained(config)
            print("Checkpoint upgraded.")
        else:
            print("Checkpoint upgrade aborted.")
            exit(1)

import torch
def print_tensor_shape_recursive(var_name, inputs_embeds, level=0):
    # return # do nothing for non-degug mode
    if isinstance(inputs_embeds, torch.Tensor):
        print(f"{'  ' * level} {var_name}: Tensor Shape: {inputs_embeds.shape}")
    elif isinstance(inputs_embeds, list):
        for i, element in enumerate(inputs_embeds):
            if isinstance(element, (int, str)):
                print(f"{'  ' * (level + 1)} {var_name}[{i}]: {type(element)}")
            else:
                print_tensor_shape_recursive(f"{var_name}[{i}]", element, level + 1)
    else:
        print(f"{'  ' * level}Unsupported inputs_embeds type for {var_name}'s type {type(inputs_embeds)}.")