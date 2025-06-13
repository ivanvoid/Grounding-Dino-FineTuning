# tests.py
import unittest

class TestModel(unittest.TestCase):
    def test_0(self):
        from config import ConfigurationManager
        from train import setup_model
        import numpy as np

        config_path = 'configs/train_config.yaml'

        data_config, model_config, training_config = ConfigurationManager.load_config(config_path)

        use_lora = training_config.use_lora
        # use_lora = False

        model = setup_model(
            model_config, 
            use_lora, 
            training_config.lora_rank,
            'cpu')
        model.eval()
        
        var = np.sum([x.sum().item() for x in model.parameters()])
        print("Sum of final model: ",var)
    
    def test_1(self):
        model_0 = 14952.569755509146
        model_1 = 14951.771855017927
        model_2 = 14950.811651024327
        model_3 = 14950.543093225278
        model_4 = 14950.34334634227
        model_5 = 14950.322413557034

unittest.main()
