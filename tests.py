# tests.py
import unittest
"""
- Model is setup
- Lora is added
- is model is frozen if LORA is False?
"""

class TestModel(unittest.TestCase):
    def test_0(self):
        pass
        # from config import ConfigurationManager
        # from finetune import setup_model
        # import numpy as np

        # config_path = 'configs/train_config.yaml'

        # data_config, model_config, training_config = ConfigurationManager.load_config(config_path)

        # use_lora = training_config.use_lora
        # # use_lora = False

        # model = setup_model(
        #     model_config, 
        #     use_lora, 
        #     training_config.lora_rank,
        #     'cpu')
        # model.eval()
        
        # var = np.sum([x.sum().item() for x in model.parameters()])
        # print("Sum of final model: ",var)
    
    def test_load_model(self):
        return
        from config import ConfigurationManager
        from groundingdino.util.train import load_model
        
        config_path = 'configs/train_config.yaml'
        all_cfgs = ConfigurationManager.load_config(config_path)
        data_config, model_config, training_config = all_cfgs

        model = load_model(
            model_config.config_path,
            model_config.weights_path,
            use_lora=model_config.use_lora,
            lora_rank=model_config.lora_rank,
            verbose=True)

        # self.assertTrue()

    def test_add_lora(self):
        import torch
        torch.manual_seed(0)
        # Create new model
        from config import ConfigurationManager, DataConfig, ModelConfig
        from finetune import setup_model 
        config_path = 'configs/train_config.yaml'
        all_cfgs = ConfigurationManager.load_config(config_path)
        data_config, model_config, training_config = all_cfgs
        model = setup_model(model_config) # setup with loras


        # Loading LORA
        from groundingdino.util.model_utils import freeze_model_layers, print_frozen_status
        from groundingdino.util.lora import verify_only_lora_trainable
        if not model_config.use_lora:
            print("Freezing most of model except few layers!")
            freeze_model_layers(model)
        else:
            print( f"Is only Lora trainable?  {verify_only_lora_trainable(model)} ")
        print_frozen_status(model)


        # Define trainer class
        from finetune import setup_data_loaders, GroundingDINOTrainer
        train_loader, val_loader = setup_data_loaders(data_config)
        steps_per_epoch = len(train_loader.dataset) // data_config.batch_size
        trainer = GroundingDINOTrainer(
            model,
            num_steps_per_epoch=steps_per_epoch,
            num_epochs=1,
            learning_rate=1,
            use_lora=True
        )


        # Check weights
        weights_sum = trainer.save_checkpoint('debug/weight_before_training.pth', 0, 0, True, debug=True)
        self.assertEqual(weights_sum, 14959.414473279961)
        

        # Train one epoch
        for _ in range(training_config.num_epochs):
            for _, batch in enumerate(train_loader):
                losses = trainer.train_step(batch)
                break
            break
        # Check weights after training
        weights_sum = trainer.save_checkpoint('debug/weight_after_training.pth', 1, 0, True, debug=True)
        self.assertEqual(int(weights_sum), 15006)


        # Check weights with no LORA
        import numpy as np
        from groundingdino.util.inference import load_model
        
        model = load_model(model_config, 'debug/weight_before_training.pth', use_lora=True)
        weights_sum = np.sum([x.sum().item() for x in model.parameters()])
        print("Loaded weights fresh LORA: ", weights_sum)
        self.assertEqual(weights_sum, 14865.556169376534)
        
        model.eval()
        weights_sum = np.sum([x.sum().item() for x in model.parameters()])
        print("Loaded weights eval: ", weights_sum)
        self.assertEqual(weights_sum, 14865.556169376534)
        
        model = load_model(model_config, use_lora=False)
        weights_sum = np.sum([x.sum().item() for x in model.parameters()])
        print("Loaded weights no-LORA: ", weights_sum)
        self.assertEqual(weights_sum, 14865.556169376534)

        # Check weights with LORA - not merged
        model = load_model(model_config, 'debug/weight_after_training.pth', use_lora=True, merge_lora=False)
        weights_sum = np.sum([x.sum().item() for x in model.parameters()])
        print("Loaded weights LORA non-merged: ", weights_sum)
        self.assertEqual(int(weights_sum), 15006)

        # Check weights with LORA - merged
        model = load_model(model_config, 'debug/weight_after_training.pth', use_lora=True, merge_lora=True)
        weights_sum = np.sum([x.sum().item() for x in model.parameters()])
        print("Loaded weights LORA merged: ", weights_sum)
        self.assertEqual(int(weights_sum), int(14835))

unittest.main()
