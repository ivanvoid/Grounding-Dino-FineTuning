import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torchvision.ops as ops
from torchvision.ops import box_convert

from groundingdino.util.inference import load_model, load_image, predict, annotate
from groundingdino.util.inference import GroundingDINOVisualizer
from config import ConfigurationManager, DataConfig, ModelConfig

def apply_nms_per_phrase(image_source, boxes, logits, phrases, threshold=0.3):
    h, w, _ = image_source.shape
    scaled_boxes = boxes * torch.Tensor([w, h, w, h])
    scaled_boxes = box_convert(boxes=scaled_boxes, in_fmt="cxcywh", out_fmt="xyxy")
    nms_boxes_list, nms_logits_list, nms_phrases_list = [], [], []

    print(f"The unique detected phrases are {set(phrases)}")

    for unique_phrase in set(phrases):
        indices = [i for i, phrase in enumerate(phrases) if phrase == unique_phrase]
        phrase_scaled_boxes = scaled_boxes[indices]
        phrase_boxes = boxes[indices]
        phrase_logits = logits[indices]

        keep_indices = ops.nms(phrase_scaled_boxes, phrase_logits, threshold)
        nms_boxes_list.extend(phrase_boxes[keep_indices])
        nms_logits_list.extend(phrase_logits[keep_indices])
        nms_phrases_list.extend([unique_phrase] * len(keep_indices))

    return torch.stack(nms_boxes_list), torch.stack(nms_logits_list), nms_phrases_list

def get_metrics(labels, bboxes, results, label_map):
    
    true_labels = sorted(labels)
    # pred_labels = sorted([l.replace('.','').split(' ')[0] for l in results['phrases']])
    pred_labels = sorted([' '.join(l.replace('.','').split(' ')[:-1]) for l in results['phrases']])

    try:
        true_labels = [label_map[l] for l in true_labels]
        pred_labels = [label_map[l] for l in pred_labels]
    except:
        import pdb;pdb.set_trace()

    true_labels = torch.tensor(true_labels)
    pred_labels = torch.tensor(pred_labels)

    cm = get_confusion(true_labels, pred_labels)
    
    iou_metric = get_iou(labels, bboxes, results)
    accuracy  = get_accuracy(cm)
    precision = get_precision(cm)
    recall = get_recall(cm)
    f1 = 0
    if precision + recall > 0:
        f1 = get_f1(precision, recall)
    
    result = {
        'acc': accuracy,
        'precision':precision,
        'recall':recall,
        'f1':f1,
        'iou': iou_metric,
    }

    return result

def get_confusion(true_labels, pred_labels):
    cm = {
        'tp':0,
        'tn':0,
        'fp':0,
        'fn':0,
        'total':0
    }

    # Convert tensors to numpy arrays for easier manipulation
    true_labels = true_labels.numpy()
    pred_labels = pred_labels.numpy()
    for pred in set(pred_labels):
        if pred in true_labels:
            cm['tp'] += 1
        else:
            cm['fp'] += 1
            
    # Calculate TN and FN
    for true in true_labels:
        if true not in pred_labels:
            cm['fn'] += 1
    
    # import pdb;pdb.set_trace()
    total = max(len(true_labels), len(pred_labels))
    assert cm['tn'] >= 0
    cm['total'] = total
    cm['tn'] = total - (cm['tp'] + cm['fp'] + cm['fn'])
    if cm['tn'] < 0:
        import pdb;pdb.set_trace()
    return cm

def get_accuracy(cm):
    return (cm['tp'] + cm['tn']) / cm['total']
def get_precision(cm):
    return cm['tp'] / (cm['tp'] + cm['fp']) if (cm['tp'] + cm['fp']) > 0 else 0.0
def get_recall(cm):
    return cm['tp'] / (cm['tp'] + cm['fn']) if (cm['tp'] + cm['fn']) > 0 else 0.0
def get_f1(precision, recall):
    return 2 * (precision * recall) / (precision + recall)


def get_iou(labels, bboxes, results):
    output = []

    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(results['phrases']):
            if true_label in pred_label:
                pred_box = results['boxes'][j].unsqueeze(0).float()
                
                pred_box = box_convert(pred_box, in_fmt='cxcywh', out_fmt='xyxy')

                true_box = torch.tensor(bboxes[i]).unsqueeze(0).float()
                
                true_box = box_convert(true_box, in_fmt='xywh', out_fmt='xyxy')
                

                output += [ops.box_iou(pred_box, true_box).item()]
    return output

def print_statistics(all_metrics):
    print('\n'+'*'*32)
    val = torch.tensor(all_metrics['acc'])
    print('Accuracy: {:.4f}'.format(val.sum() / len(val)) )
    
    val = torch.tensor(all_metrics['precision'])
    print('Precision: {:.4f} ± {:.4f}'.format(torch.mean(val), torch.std(val)) )

    val = torch.tensor(all_metrics['recall'])
    print('Recall: {:.4f} ± {:.4f}'.format(torch.mean(val), torch.std(val)) )

    val = torch.tensor(all_metrics['f1'])
    print('F1: {:.4f} ± {:.4f}'.format(torch.mean(val), torch.std(val)) )

    val = torch.tensor(all_metrics['iou'])
    print('IoU: {:.4f} ± {:.4f}'.format(torch.mean(val), torch.std(val)) )

    print('*'*32)

def process_images(
        model,
        text_prompt,
        data_config,
        box_threshold=0.35,
        text_threshold=0.25
):
    visualizer = GroundingDINOVisualizer(save_dir="visualizations")

    all_metrics = {
        'acc':[],
        'precision':[],
        'recall':[],
        'f1':[],
        'iou':[],
    }

    # Get bbox and imgname
    df = pd.read_csv(data_config.val_ann)
    # import pdb;pdb.set_trace()
    label_map = {l:i for i, l in enumerate(np.unique(df.label_name.values))}
    
    for imagename in tqdm(np.unique(df.image_name.values)):
        _df = df[df.image_name==imagename]
        labels = _df.label_name.values
        bboxes = np.array([_df.bbox_x, _df.bbox_y, _df.bbox_width, _df.bbox_height]).T

        image_path = os.path.join(data_config.val_dir, imagename)
        image_source, image = load_image(image_path)
        # import pdb;pdb.set_trace()
        results = visualizer.predict_image(
            model,image,text_prompt,image_source,imagename,
            box_th=box_threshold,txt_th=text_threshold)
        
        if not (results is None):
            metrics = get_metrics(labels, bboxes, results, label_map)

            # all_metrics['iou'].append(metrics['iou'])
            all_metrics['acc'].append(metrics['acc'])
            all_metrics['precision'].append(metrics['precision'])
            all_metrics['recall'].append(metrics['recall'])
            all_metrics['f1'].append(metrics['f1'])
            all_metrics['iou'] += metrics['iou']

    print_statistics(all_metrics)


if __name__ == "__main__":
    # Config file of the prediction, the model weights can be complete model weights but if use_lora is true then lora_wights should also be present see example
    
    # config file
    config_path="configs/test_config.yaml"
    # config_path="configs/custum_test_config.yaml"

    text_prompt="shirt .bag .pants"
    # text_prompt="crimpers . cutter . drill . hammer . hand file . measurement tape . pen . pendant control . pliers . power supply . scissors . screwdriver . screws . tape . tweezers . usb cable . vernier caliper . whiteboard marker . wire . wrench"
    

    data_config, model_config, training_config = ConfigurationManager.load_config(config_path)
    model = load_model(model_config,training_config.use_lora)
    process_images(model, text_prompt, data_config)

"""
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

"""