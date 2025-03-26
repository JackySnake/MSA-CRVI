import argparse, os, random
from ast import If
from unittest import result
import torch
from torch.utils.data import DataLoader

from csmv_dataset import CSMV_Dataset, CSMV_Dataset_r2plus1, CSMV_Dataset_VideoMAEv2FPS16, CSMV_Dataset_VideoMAEv2FPS24, csmv_collate_fn

from train_vccsv import train 
from train_vccsv import evaluate
import numpy as np
from utils.compute_args import compute_args
from easydict import EasyDict as eDict


from collections import OrderedDict
from typing import Dict, Optional, List, Tuple, Union,Callable
from loguru import logger
import json
import pickle
import tqdm

from config.config import model_cfg

#model 
from model_VCCSA import MVCAnalysis_Model

def read_json_file(path: str) -> Union[Dict, List[Dict],List]:
    import json
    with open(path, "r") as f:
        return json.load(f)
def write_pkl_file(file_name, data) -> None:
    logger.info(f"Writing {file_name}")
    with open(file_name, "wb") as f:
        pickle.dump(data, f)
def write_json_file(file_name, data) -> None:
    logger.info(f"Writing {file_name}")
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)

def parse_args():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--model', type=str, default="VCCSA", choices=["VCCSA"])
    parser.add_argument('--layer', type=int, default=4)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--dropout_r', type=float, default=0.1)
    parser.add_argument('--multi_head', type=int, default=8)
    parser.add_argument('--ff_size', type=int, default=2048)
    parser.add_argument('--word_embed_size', type=int, default=300)

    # Data
    parser.add_argument('--lang_seq_len', type=int, default=60)
    parser.add_argument('--audio_seq_len', type=int, default=60)
    parser.add_argument('--video_seq_len', type=int, default=60)
    parser.add_argument('--audio_feat_size', type=int, default=80)
    parser.add_argument('--video_feat_size', type=int, default=512)

    # Training
    parser.add_argument('--output', type=str, default='ckpt/')
    parser.add_argument('--name', type=str, default='exp0/')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_epoch', type=int, default=120)
    parser.add_argument('--opt', type=str, default="Adam")
    parser.add_argument('--opt_params', type=str, default="{'betas': '(0.9, 0.98)', 'eps': '1e-9'}")
    parser.add_argument('--lr_base', type=float, default=0.00001)
    parser.add_argument('--lr_decay', type=float, default=0.95)
    parser.add_argument('--lr_decay_times', type=int, default=60)
    parser.add_argument('--warmup_epoch', type=float, default=0)
    parser.add_argument('--grad_norm_clip', type=float, default=-1)
    parser.add_argument('--eval_start', type=int, default=0)
    parser.add_argument('--early_stop', type=int, default=5)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--optim', type=str, default='adma')

    # Dataset and task
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['CSMV', 'CSMV_r2plus1', 'CSMV_VideoMAEv2FPS16', 'CSMV_VideoMAEv2FPS24'],
        default='CSMV')
    # parser.add_argument('--task', type=str, choices=['sentiment', 'emotion'], default='sentiment')
    parser.add_argument('--task_binary', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=8)

    # JQ_mvc_dataset
    parser.add_argument('--video_feature_dir', type=str)
    parser.add_argument('--opinion_label_map', type=str)
    parser.add_argument('--emotion_label_map', type=str)
    parser.add_argument('--video_comment', type=str)
    parser.add_argument('--annotations', type=str)
    parser.add_argument('--train_set', type=str, default='train_set.json')
    parser.add_argument('--dev_set', type=str, default='dev_set.json')
    parser.add_argument('--test_set', type=str, default='test_set.json')
    parser.add_argument('--datadir', type=str, default='data/CSMV/commentDataset')
    # cuda GPU
    parser.add_argument('--cuda', type=str, default='0')
    # pre_trainedmodel
    parser.add_argument('--pre_trained_LM', type=str, default='/home/jac/.cache/torch/hub/transformers/roberta-base')
    # parser.add_argument('--model_path', type=str)  
    # experiment setting
    parser.add_argument('--mv_att', type=bool, default=False)
    
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--model_name',type=str,default="model_vccsv")

    args = parser.parse_args()
    return args

def get_model(mode_name:str)-> Callable:
    logger.info(f"model_name:{mode_name}")
    if mode_name == "model_vccsv":
        from model_VCCSA import MVCAnalysis_Model #
        return MVCAnalysis_Model
    else: #could use for other model design
        from model_VCCSA import MVCAnalysis_Model 
        return MVCAnalysis_Model
        
def get_model_ck(model_dir:str,last_ck=-1)->List: #get all checkpoint
    files = os.listdir(model_dir)
    model_files = [file for file in files if file.endswith(".pkl") and file.startswith("best")]
    get_epoch:callable = lambda file : int(file.split("_")[-1].split(".")[0])
    model_files_sorted= sorted(model_files,key=lambda file : get_epoch(file))
    
    if last_ck == -1:
        return model_files_sorted
    else:
        idx = min(last_ck,len(model_files))
        return model_files_sorted[-idx:]

def get_model_top_n_ck(model_dir:str,topk=-1)->List: # get top performance checkpoint
    files = os.listdir(model_dir)
    model_files = [file for file in files if file.endswith(".pkl") and file.startswith("best")]
    get_dev_score:callable = lambda file : float(file.split("_")[1])
    model_files_sorted = sorted(model_files,key=lambda file : get_dev_score(file),reverse=True)
    
    if topk == -1:
        return model_files_sorted
    else:
        idx = min(topk,len(model_files))
        return model_files_sorted[:idx]


# --model_path need, direct to pkl file
if __name__ == '__main__':
    # Base on args given, compute new args
    args = compute_args(parse_args())
    
    logger.add(os.path.join(args.output,f"{args.name}_eval.log"))

    eval_dset = eval(args.dataloader)('valid', args, dataroot=args.datadir)
    test_dset = eval(args.dataloader)('test', args, dataroot=args.datadir)

    eval_loader = DataLoader(eval_dset,
                             args.batch_size,
                             num_workers=args.num_workers,
                             pin_memory=True,
                             collate_fn=csmv_collate_fn)
    test_loader = DataLoader(test_dset,
                            args.batch_size,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            collate_fn=csmv_collate_fn)

    logger.info(f"model name:{args.model_name}")
    MVCAnalysis_Model = get_model(args.model_name)
    net = MVCAnalysis_Model(args, eDict(model_cfg))
    # Create Checkpoint dir
    ck_dir = os.path.join(args.output, args.name, 'evaluate')
    if not os.path.exists(ck_dir):
        os.makedirs(ck_dir)

    model_files = get_model_top_n_ck(model_dir=args.model_dir,topk=1) # get best model on eval

    for file in tqdm.tqdm(model_files):
        logger.info(f"check file:{file}")
        epoch = file.split("_")[-1].split(".")[0]
        logger.info(f"current epoch_{epoch} staring eavl")
        model_ck = os.path.join(args.model_dir,file)
        if not os.path.exists(model_ck):
            logger.error("The model path is error.")
        
        ck_state_dict = torch.load(model_ck,map_location='cpu').get('state_dict')
        state_dict = OrderedDict()
        [state_dict.update({".".join(k.split(".")[1:]) : v}) for k,v in ck_state_dict.items()]
        net.load_state_dict(state_dict)
        net = net.cuda('cuda:'+str(args.cuda))

        # Run evaluate
        net.eval()

        ## val dataset 
        eval_accuracies, result = evaluate(net, eval_loader, args) #eval dataloader 

        logger.info(f"dev_performance_model:{eval_accuracies}")
        write_json_file(os.path.join(ck_dir,f"eoch{epoch}_dev_predict_model.json"), result)
        write_json_file(os.path.join(ck_dir,f"eoch{epoch}_dev_performance_model.json"), eval_accuracies)
        
        ## test dataset 
        test_accuracies, result = evaluate(net, test_loader, args) #test dataloader 

        logger.info(f"test_performance_model:{test_accuracies}")
        write_json_file(os.path.join(ck_dir,f"eoch{epoch}_test_predict_model.json"), result)
        write_json_file(os.path.join(ck_dir,f"eoch{epoch}_test_performance_model.json"), test_accuracies)

    logger.warning("over")

