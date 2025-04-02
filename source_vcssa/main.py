import argparse, os, random
import imp
from xmlrpc.client import boolean
import torch
from torch.utils.data import DataLoader
import numpy as np
from easydict import EasyDict as eDict
# dataset load
from csmv_dataset import CSMV_Dataset, CSMV_Dataset_r2plus1, CSMV_Dataset_VideoMAEv2FPS16, CSMV_Dataset_VideoMAEv2FPS24, csmv_collate_fn
# traning and evaluate
from train_vccsv import train
# model and training setting
from utils.compute_args import compute_args

from transformers import (AdamW, get_linear_schedule_with_warmup)


from config.config import model_cfg # defalut

from model_VCCSA import MVCAnalysis_Model # add dropout and padding dealing

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
    # AdamW
    parser.add_argument('--warmup_proportion', type=float, default=0.1)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    # Dataset and task
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['CSMV', 'CSMV_r2plus1', 'CSMV_VideoMAEv2FPS24'],
        default='CSMV')
    parser.add_argument('--task', type=str, choices=['sentiment', 'emotion'], default='sentiment')
    parser.add_argument('--task_binary', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=8)

    # CSMV
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
    parser.add_argument('--pre_trained_LM',
                        type=str,
                        default='~/.cache/torch/hub/transformers/roberta-base')

    # experiment setting
    parser.add_argument('--mv_att', type=bool, default=False)

    parser.add_argument('--fine_ck_path', type=str, default=None)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Base on args given, compute new args

    args = compute_args(parse_args())
    # different visual feature
    if args.dataset == "CSMV":
        from config.config import model_cfg #
    elif args.dataset == "CSMV_r2plus1": #could use for other model design
        from config.config_r21d import model_cfg  
    elif args.dataset == "CSMV_VideoMAEv2FPS24": #could use for other model design
        from config.config_VideoMAEv2fps24 import model_cfg 
    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # DataLoader
    train_dset = eval(args.dataloader)(
        'train', args, dataroot=args.datadir)  # to get the target dataset code with eval, the function in the bucket
    eval_dset = eval(args.dataloader)('valid', args, dataroot=args.datadir)
    train_loader = DataLoader(train_dset,
                              args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              collate_fn=csmv_collate_fn)
    # train_loader = DataLoader(train_dset, args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    eval_loader = DataLoader(eval_dset,
                             args.batch_size,
                             num_workers=args.num_workers,
                             pin_memory=True,
                             collate_fn=csmv_collate_fn)

    # Net
    # net = eval(args.model)(args, train_dset.vocab_size, train_dset.pretrained_emb).cuda()
    # build the model
    net = MVCAnalysis_Model(args, eDict(model_cfg))
    if args.fine_ck_path:
        print(f"load ck :{args.fine_ck_path}")
        ck_state_dict = torch.load(args.fine_ck_path, map_location='cpu').get('state_dict')
        from collections import OrderedDict
        state_dict = OrderedDict()
        [state_dict.update({".".join(k.split(".")[1:]): v}) for k, v in ck_state_dict.items()]
        net.load_state_dict(state_dict)

    print("Total number of parameters : " + str(sum([p.numel() for p in net.parameters()]) / 1e6) + "M")
    # net = net.cuda()
    net = net.cuda('cuda:' + str(args.cuda))

    # AdamW
    t_total = len(train_loader) * args.max_epoch
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{
        'params': [p for n, p in net.named_parameters() if not any(nd in n for nd in no_decay)],
        'weight_decay': args.weight_decay
    }, {
        'params': [p for n, p in net.named_parameters() if any(nd in n for nd in no_decay)],
        'weight_decay': 0.0
    }]
    optim = AdamW(optimizer_grouped_parameters, lr=args.lr_base, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optim,
                                                num_warmup_steps=int(t_total * args.warmup_proportion),
                                                num_training_steps=t_total)

    # Create Checkpoint dir
    if not os.path.exists(os.path.join(args.output, args.name)):
        os.makedirs(os.path.join(args.output, args.name))

    # Run training
    # eval_accuracies = jq_train(net, train_loader, eval_loader, args) #train dataloader, eval dataloader
    eval_accuracies = train(net, train_loader, eval_loader, optim, args,
                               scheduler)  #train dataloader, eval dataloader
