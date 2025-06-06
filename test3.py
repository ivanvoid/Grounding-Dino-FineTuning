import torch
import numpy as np
from tqdm import tqdm

from config import ConfigurationManager, DataConfig, ModelConfig
from groundingdino.util.inference import load_model, load_image, predict, annotate
from train import setup_data_loaders, GroundingDINOTrainer
from groundingdino.util.inference import GroundingDINOVisualizer

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

    visualizer = GroundingDINOVisualizer(save_dir='visualizations')

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

        epoch = 0
        visualizer.visualize_epoch(model, val_loader, epoch, trainer.prepare_batch, box_th=0.3, txt_th= 0.2)


    for k,v in metrics.items():
        print(f'{k} : {np.mean(v)} ± {np.std(v)}')


def main():
    # config_path="configs/test_config.yaml"
    # text_prompt="shirt .bag .pants"
    

    # config_path="configs/custum_test_config.yaml"
    # text_prompt="crimpers . cutter . drill . hammer . hand file . measurement tape . pen . pendant control . pliers . power supply . scissors . screwdriver . screws . tape . tweezers . usb cable . vernier caliper . whiteboard marker . wire . wrench"

    config_path="configs/tiny_config.yaml"
    text_prompt="wire. power supply. connector. terminal."


    data_config, model_config, training_config = ConfigurationManager.load_config(config_path)
    model = load_model(model_config,training_config.use_lora)
    
    validate(model, text_prompt, data_config)
    

if __name__ == "__main__":
    main()
