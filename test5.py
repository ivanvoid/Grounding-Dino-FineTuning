import torch
import numpy as np
from tqdm import tqdm

from config import ConfigurationManager, DataConfig, ModelConfig
from groundingdino.util.inference import load_model, load_image, predict, annotate
from train import setup_data_loaders, GroundingDINOTrainer
from groundingdino.util.inference import GroundingDINOVisualizer

import pycocotools
import torch.nn as nn
from torchvision.ops.boxes import nms
from groundingdino.util import box_ops, get_tokenlizer

from setproctitle import setproctitle
setproctitle("G-DINO-Eval")

def create_positive_map(tokenized, tokens_positive,cat_list,caption):
    """construct a map such that positive_map[i,j] = True iff box i is associated to token j"""
    positive_map = torch.zeros((len(tokens_positive), 256), dtype=torch.float)

    for j,label in enumerate(tokens_positive):

        start_ind = caption.find(cat_list[label])
        end_ind = start_ind + len(cat_list[label]) - 1
        beg_pos = tokenized.char_to_token(start_ind)
        try:
            end_pos = tokenized.char_to_token(end_ind)
        except:
            end_pos = None
        if end_pos is None:
            try:
                end_pos = tokenized.char_to_token(end_ind - 1)
                if end_pos is None:
                    end_pos = tokenized.char_to_token(end_ind - 2)
            except:
                end_pos = None
        # except Exception as e:
        #     print("beg:", beg, "end:", end)
        #     print("token_positive:", tokens_positive)
        #     # print("beg_pos:", beg_pos, "end_pos:", end_pos)
        #     raise e
        # if beg_pos is None:
        #     try:
        #         beg_pos = tokenized.char_to_token(beg + 1)
        #         if beg_pos is None:
        #             beg_pos = tokenized.char_to_token(beg + 2)
        #     except:
        #         beg_pos = None
        # if end_pos is None:
        #     try:
        #         end_pos = tokenized.char_to_token(end - 2)
        #         if end_pos is None:
        #             end_pos = tokenized.char_to_token(end - 3)
        #     except:
        #         end_pos = None
        if beg_pos is None or end_pos is None:
            continue
        if beg_pos < 0 or end_pos < 0:
            continue
        if beg_pos > end_pos:
            continue
        # assert beg_pos is not None and end_pos is not None
        positive_map[j,beg_pos: end_pos + 1].fill_(1)
    return positive_map 


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, num_select=100,text_encoder_type='text_encoder_type', nms_iou_threshold=-1,args=None) -> None:
        super().__init__()
        self.num_select = num_select
        self.tokenizer = get_tokenlizer.get_tokenlizer(text_encoder_type)
        if args.use_coco_eval:
            from pycocotools.coco import COCO
            coco = COCO(args.coco_val_path)
            category_dict = coco.loadCats(coco.getCatIds())
            cat_list = [item['name'] for item in category_dict]
        else:
            cat_list=args.label_list
        caption = " . ".join(cat_list) + ' .'
        tokenized = self.tokenizer(caption, padding="longest", return_tensors="pt")
        label_list = torch.arange(len(cat_list))
        pos_map=create_positive_map(tokenized,label_list,cat_list,caption)
        # build a mapping from label_id to pos_map
        if args.use_coco_eval:
            id_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31, 27: 32, 28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44, 40: 46,
                    41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55, 50: 56, 51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65, 60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90}
            import pdb;pdb.set_trace()
            new_pos_map = torch.zeros((91, 256))
            for k, v in id_map.items():
                new_pos_map[v] = pos_map[k]
            pos_map=new_pos_map


        self.nms_iou_threshold=nms_iou_threshold
        self.positive_map = pos_map

    @torch.no_grad()
    def forward(self, outputs, target_sizes, not_to_xyxy=False, test=False):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        num_select = self.num_select
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']


        prob_to_token = out_logits.sigmoid()
        pos_maps = self.positive_map.to(prob_to_token.device)
        for label_ind in range(len(pos_maps)):
            if pos_maps[label_ind].sum() != 0:
                pos_maps[label_ind]=pos_maps[label_ind]/pos_maps[label_ind].sum()

        prob_to_label = prob_to_token @ pos_maps.T

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = prob_to_label
        topk_values, topk_indexes = torch.topk(prob.view(prob.shape[0], -1), num_select, dim=1)
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, prob.shape[2], rounding_mode='trunc')
        labels = topk_indexes % prob.shape[2]
        if not_to_xyxy:
            boxes = out_bbox
        else:
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        # if test:
        #     assert not not_to_xyxy
        #     boxes[:,:,2:] = boxes[:,:,2:] - boxes[:,:,:2]
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        if self.nms_iou_threshold > 0:
            item_indices = [nms(b, s, iou_threshold=self.nms_iou_threshold) for b,s in zip(boxes, scores)]

            results = [{'scores': s[i], 'labels': l[i], 'boxes': b[i]} for s, l, b, i in zip(scores, labels, boxes, item_indices)]
        else:
            results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        return results


def validate(model, captions, data_config):
    model.eval()

    _, val_loader = setup_data_loaders(data_config)

    trainer = GroundingDINOTrainer(
        model,
        num_steps_per_epoch=1,
        num_epochs=1,
        warmup_epochs=1,
        learning_rate=0.1,
        use_lora=True
    )
    """
    criterion = SetCriterion(matcher=matcher, weight_dict=weight_dict,
                             focal_alpha=args.focal_alpha, focal_gamma=args.focal_gamma,losses=losses
                             )
    criterion.to(device)
    postprocessors = {'bbox': PostProcess(num_select=args.num_select  , text_encoder_type=args.text_encoder_type,nms_iou_threshold=args.nms_iou_threshold,args=args)}
    """
    from types import SimpleNamespace
    # coco_dir = './multimodal-data/dataset_coco/'
    # args = {
    #     'use_coco_eval':False,
    #     'coco_val_path':'./multimodal-data/dataset_coco/annotations.json',
    #     'label_list': ['bag', 'shirt', 'bag']
    # }
    coco_dir = './multimodal-data/GAI20II_COCO/'
    args = {
        'use_coco_eval':False,
        'coco_val_path':'./multimodal-data/GAI20II_COCO/annotations.json',
        'label_list': ['crimpers', 'cutter', 'drill', 'hammer', 'hand file', 'measurement tape', 'pen', 'pendant control', 'pliers', 'power supply', 'scissors', 'screwdriver', 'screws', 'tape', 'tweezers', 'usb cable', 'vernier caliper', 'whiteboard marker', 'wire', 'wrench']
    }


    args = SimpleNamespace(**args)

    postprocessors = {'bbox': PostProcess(num_select=100,text_encoder_type='bert-base-uncased', nms_iou_threshold=0.5,args=args)}

    coco_stats = trainer.evaluate(
        model=model,
        criterion=trainer.criterion,
        postprocessors=postprocessors,
        data_loader=val_loader,
        base_ds=coco_dir, # coco dataset basedir
        device='cuda',
        output_dir='./output_eval_coco_dir',
        wo_class_error=False,
        args=args,
        logger=None,
        verbose=True)
    return coco_stats



def main():
    # models_path = './weights/20250610_1606'
    # models_path = './weights/20250610_1745' 
    models_path = './weights/20250611_1559' 

    import os
    model_names = [name for name in os.listdir(models_path) if '.pth' in name]

    config_path="configs/test_config.yaml"
    text_prompt="shirt .bag .pants"
    
    # config_path="configs/custum_test_config.yaml"
    # text_prompt="crimpers . cutter . drill . hammer . hand file . measurement tape . pen . pendant control . pliers . power supply . scissors . screwdriver . screws . tape . tweezers . usb cable . vernier caliper . whiteboard marker . wire . wrench"
    
    data_config, model_config, training_config = ConfigurationManager.load_config(config_path)
    statistic = []
    
    model_names = sorted(model_names, key=lambda x: int(x[:-4].split('_')[-1]))

    for model_name in model_names:
        print(f'Evaluating {model_name}')

        model_config = ModelConfig.from_dict({
            "config_path": model_config.config_path,
            "weights_path": model_config.weights_path,
            "lora_weights": os.path.join(models_path, model_name)
        })
        print(model_config)

        model = load_model(model_config, training_config.use_lora, lora_rank=training_config.lora_rank)
        # model = load_model(model_config, False, lora_rank=training_config.lora_rank)
    
        coco_stats = validate(model, text_prompt, data_config)
        statistic += [[coco_stats]]
        print('statistic:\n',statistic)

        os.remove('pred_coco.json')
    
    statistic = np.concat(statistic, axis=0)
    np.savetxt('eval_statistic.csv', statistic, delimiter=',') 

if __name__ == "__main__":
    main()
