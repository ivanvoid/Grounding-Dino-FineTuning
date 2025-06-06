import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

import os
import sys
import time
import torch

from config import ConfigurationManager
from groundingdino.util.inference import GroundingDINOVisualizer
from groundingdino.util.inference import load_model, load_image

def predict(model, data_in):
    visualizer = GroundingDINOVisualizer(save_dir="visualizations")
    caption = data_in['caption']
    for path in data_in['image_path']:
        image_source, image = load_image(path)

        try:
            t1 = time.time()
            with torch.no_grad():
                results = visualizer.visualize_image(
                    model,
                    image,
                    caption,
                    image_source,
                    f'{time.time()}.png',
                    box_th=0.35, # 25
                    txt_th=0.25) # 15
            t2 = time.time()
            print(f'Inference time: {t2-t1:.1f} sec')
        except:
            print('Exit.')
    return -1

import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description="A brief description of your program.")
    
    # Add arguments
    parser.add_argument(
        '--img_folder', 
        type=str, 
        required=True, 
        help='Description for argument 1'
    )
    
    parser.add_argument(
        '--prompt', 
        type=str, 
        # default='', 
        help='Description for argument 2'
    )

    args = parser.parse_args()
    return args

"""
python inference.py --img_folder /home/ivan/data/sim --prompt "red wire. connector. power supply."
"""

def main():
    args = arg_parse()
    print(args)

    # Define data

    base = args.img_folder # 'multimodal-data/our_data/'
    paths = [os.path.join(base,name) for name in sorted(os.listdir(base))]

    caption = args.prompt #sys.argv[1]
    print(f"\nCaption: {caption}\n")

    data_in = {
        'image_path': paths,
        'caption': caption
    }

    # config_path = 'configs/test_config.yaml'
    config_path = 'configs/tiny_config.yaml'

    # Define model
    data_config, model_config, training_config = ConfigurationManager.load_config(config_path)
    model = load_model(model_config,training_config.use_lora)
    model.eval()

    data_out = predict(model, data_in)


if __name__ == "__main__":
    main()
