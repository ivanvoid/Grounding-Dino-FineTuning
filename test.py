from config import ConfigurationManager, DataConfig, ModelConfig
import os
import yaml
import torch
from groundingdino.util.train import load_model

def setup_model(model_config: ModelConfig, use_lora: bool=False, lora_rank:int=32) -> torch.nn.Module:
    return load_model(
        model_config.config_path,
        model_config.weights_path,
        use_lora=use_lora,
        lora_rank=lora_rank
    ).to('cuda:0')

config_path = 'configs/train_config.yaml'

data_config, model_config, training_config = ConfigurationManager.load_config(config_path)

model = setup_model(model_config, training_config.use_lora, training_config.lora_rank)