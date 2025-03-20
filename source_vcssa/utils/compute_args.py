import torch


def compute_args(args):
    # DataLoader
    # if not hasattr(args, 'dataset'):  # fix for previous version
    #     args.dataset = 'MOSEI'

    # if args.dataset == "MOSEI":
    #     args.dataloader = 'Mosei_Dataset'
    # if args.dataset == "MELD":
    #     args.dataloader = 'Meld_Dataset'
    # if args.dataset == "JQ_MVC":
    #     args.dataloader = "JQ_MVC_Dataset" # set the dataset class in use
    if args.dataset == "CSMV":
        args.dataloader = 'CSMV_Dataset'
    if args.dataset == "CSMV_VideoMAEv2FPS16":
        args.dataloader = 'CSMV_Dataset_VideoMAEv2FPS16'
    if args.dataset == "CSMV_VideoMAEv2FPS24":
        args.dataloader = "CSMV_Dataset_VideoMAEv2FPS24" # set the dataset class in use
    # Loss function to use
    # if args.dataset == 'MOSEI' and args.task == 'sentiment': args.loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    # if args.dataset == 'MOSEI' and args.task == 'emotion': args.loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")
    # if args.dataset == 'MELD': args.loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

    # # TODO
    # if args.dataset == 'JQ_MVC':
    #     args.loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")
    if args.dataset == 'CSMV':
        args.loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")
    if args.dataset == 'CSMV_VideoMAEv2FPS16':
        args.loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")
    if args.dataset == 'CSMV_VideoMAEv2FPS24':
        args.loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")

    # Answer size
    # if args.dataset == 'MOSEI' and args.task == "sentiment": args.ans_size = 7
    # if args.dataset == 'MOSEI' and args.task == "sentiment" and args.task_binary: args.ans_size = 2
    # if args.dataset == 'MOSEI' and args.task == "emotion": args.ans_size = 6
    # if args.dataset == 'MELD' and args.task == "emotion": args.ans_size = 7
    # if args.dataset == 'MELD' and args.task == "sentiment": args.ans_size = 3

    # TODO
    if args.dataset == 'CSMV':
        args.ans_size = 7

    # if args.dataset == 'MOSEI': args.pred_func = "amax"
    # if args.dataset == 'MOSEI' and args.task == "emotion": args.pred_func = "multi_label"
    # if args.dataset == 'MELD': args.pred_func = "amax"

    return args
