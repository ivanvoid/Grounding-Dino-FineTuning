import torch
import numpy as np
from tqdm import tqdm

from config import ConfigurationManager, DataConfig, ModelConfig
from groundingdino.util.inference import load_model, load_image, predict, annotate
from train import setup_data_loaders, GroundingDINOTrainer
from groundingdino.util.inference import GroundingDINOVisualizer

from setproctitle import setproctitle
setproctitle("G-DINO-FineTune")

def compute_precision_recall(yTrue, predScores, thresholds):
    import sklearn.metrics
    precisions = []
    recalls = []
    # loop over each threshold from 0.2 to 0.65
    for threshold in thresholds:
        # yPred is dog if prediction score greater than threshold
        # else cat if prediction score less than threshold
        yPred = [
            "dog" if score >= threshold else "cat"
            for score in predScores
        ]
  
        # compute precision and recall for each threshold
        precision = sklearn.metrics.precision_score(y_true=yTrue,
            y_pred=yPred, average=None)
        recall = sklearn.metrics.recall_score(y_true=yTrue,
            y_pred=yPred, average=None)
  
        # append precision and recall for each threshold to
        # precisions and recalls list
        precisions.append(np.round(precision, 3))
        recalls.append(np.round(recall, 3))
    # return them to calling function
    return precisions, recalls

def mAP(model, visualizer, outputs, targets):
    import sklearn.metrics
    y_true = targets[0]['str_cls_lst']

    thresholds = np.arange(start=0.1, stop=0.7, step=0.05)

    pred_logits = outputs["pred_logits"][0].cpu().sigmoid()
    scores = pred_logits.max(dim=1)[0]

    pred_boxes = outputs["pred_boxes"][0].cpu()
    
    tokenized = outputs['tokenized']

    for threshold in thresholds:
        box_th = threshold
        txt_th = threshold
        mask = scores > box_th

        filtered_boxes = pred_boxes[mask]
        filtered_logits = pred_logits[mask]

        phrases = visualizer.extract_phrases(filtered_logits, tokenized, model.tokenizer, text_threshold=txt_th, ret_list=True)

        pred_labels = [p[0] for p in phrases]
        pred_scores = [p[1] for p in phrases]

        tp=0
        tn=0
        fp=0
        fn=0

        # precision = sklearn.metrics.precision_score(y_true=y_true,
        #     y_pred=yPred, average=None)
        # recall = sklearn.metrics.recall_score(y_true=y_true,
        #     y_pred=yPred, average=None)
        # pass


    """
    box_th, txt_th = 0.25, 0.2

    pred_logits = outputs["pred_logits"][0].cpu().sigmoid()
    scores = pred_logits.max(dim=1)[0]
    
    pred_boxes = outputs["pred_boxes"][0].cpu()
    
    mask = scores > box_th
    print(mask.sum())
    
    filtered_boxes = pred_boxes[mask];filtered_logits = pred_logits[mask]

    tokenized = outputs['tokenized']
    phrases = visualizer.extract_phrases(filtered_logits, tokenized, model.tokenizer, text_threshold=txt_th, ret_list=True)

    [p[0] for p in phrases]
    """
    import pdb;pdb.set_trace()
    print()
    pass

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
            
            
            # mAP(model, visualizer, outputs, targets)
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
