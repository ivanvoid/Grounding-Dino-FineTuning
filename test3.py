import torch
import numpy as np
from tqdm import tqdm

from config import ConfigurationManager, DataConfig, ModelConfig
from groundingdino.util.inference import load_model, load_image, predict, annotate
from train import setup_data_loaders, GroundingDINOTrainer

def validate(model, captions, data_config):

    _, val_loader = setup_data_loaders(data_config)

    trainer = GroundingDINOTrainer(
        model,
        num_steps_per_epoch=1,
        num_epochs=1,
        warmup_epochs=1,
        learning_rate=0.1,
        use_lora=True
    )

    model.eval()
    val_losses = {}
    num_batches = 0
    
    metrics = {
        'macc':[],
        'umacc':[],
        'giou':[]
    }

    with torch.no_grad():
        for batch in tqdm(val_loader):
            images, targets, captions = trainer.prepare_batch(batch)
            outputs = model(images, captions=captions)
            
            # Calculate losses
            loss_dict = trainer.criterion(
                outputs, targets, 
                captions=captions, 
                tokenizer=trainer.model.tokenizer)
            
            pred_acc = trainer.criterion.get_accuracy(outputs, targets)

            metrics['macc'] += [pred_acc['Matched_Token_Accuracy'].item()]
            metrics['umacc'] += [pred_acc['UnMatched_Token_Accuracy'].item()]
            metrics['giou'] += [1 - loss_dict['loss_giou'].item()]

    for k,v in metrics.items():
        print(f'{k} : {np.mean(v)} ± {np.std(v)}')

            # import pdb;pdb.set_trace()
            # print()

            # # Accumulate losses
            # for k, v in loss_dict.items():
            #     val_losses[k] += v.item()
                
            # val_losses['total_loss'] += sum(
            #     loss_dict[k] * self.weights_dict[k] 
            #     for k in loss_dict.keys() if k in self.weights_dict_loss).item()
            # num_batches += 1

            # # Average losses
            # return {k: v/num_batches for k, v in val_losses.items()}


    pass

def main():
    # config_path="configs/test_config.yaml"
    # text_prompt="shirt .bag .pants"
    

    config_path="configs/custum_test_config.yaml"
    text_prompt="crimpers . cutter . drill . hammer . hand file . measurement tape . pen . pendant control . pliers . power supply . scissors . screwdriver . screws . tape . tweezers . usb cable . vernier caliper . whiteboard marker . wire . wrench"



    data_config, model_config, training_config = ConfigurationManager.load_config(config_path)
    model = load_model(model_config,training_config.use_lora)
    
    validate(model, text_prompt, data_config)
    

if __name__ == "__main__":
    main()
