from __future__ import print_function
import os
import pickle
import numpy as np
import torch
# from utils.plot import plot
from utils.tokenize import tokenize, create_dict, sent_to_ix, cmumosei_2, cmumosei_7, pad_feature
from torch.utils.data import Dataset
from transformers import RobertaTokenizerFast
from typing import Dict, Optional, List, Tuple, Union
import random


def csmv_collate_fn(batch: List) -> Dict:

    def collate_comment(comment: List[Dict]):
        handle_key = ["emotion_label", "opinion_label"]
        output = {}
        for key in comment[0].keys():
            if key in handle_key:
                output[key] = torch.vstack([d.get(key) for d in comment])
            else:
                output[key] = [d.get(key) for d in comment]
        return output

    bt_output = {} 
    for bt in batch:
        for k, v in bt.items():
            if k in bt_output:
                bt_output[k].append(v)
            else:
                bt_output[k] = [v]

    # bt_output['commentkey'] = 
    bt_output['video_feat'] = torch.from_numpy(np.array(bt_output.pop('video_feat')))
    bt_output['video_feat_mask'] = torch.stack(bt_output.pop('video_feat_mask'))
    bt_output['comment_info'] = collate_comment(bt_output.pop("comment_info"))
    bt_output['other_comment_info'] = collate_comment(bt_output.pop("other_comment_info"))

    return bt_output


class CSMV_Dataset(Dataset):
    def __init__(self, name, args, token_to_ix=None, dataroot='data'):
        super(CSMV_Dataset, self).__init__()
        assert name in ['train', 'valid', 'test', 'private']
        self.name = name
        self.args = args
        self.private_set = name == 'private'
        self.dataroot = dataroot
        self.video_feature_dir = args.video_feature_dir

        self.emotion_label_map = self.read_json_file(os.path.join(self.dataroot, args.emotion_label_map)) #label mapping file of emotion
        self.opinion_label_map = self.read_json_file(os.path.join(self.dataroot, args.opinion_label_map)) #label mapping file of opinion
        # self.video_to_comment = self.read_pkl_file(os.path.join(self.dataroot, args.video_comment)) # input data, FOR resampling
        self.video_to_comment = self.read_json_file(os.path.join(self.dataroot, args.video_comment)) # input data, FOR resampling
        self.annotations = self.read_json_file(os.path.join(self.dataroot, args.annotations)) # annotation of input data, label data
        if name == 'train': # input data, get the commentkeys which use
            self.label_data_select = self.read_json_file(os.path.join(self.dataroot, args.train_set))
        elif name == 'valid':
            self.label_data_select = self.read_json_file(os.path.join(self.dataroot, args.dev_set))
        elif name == 'test':
            self.label_data_select = self.read_json_file(os.path.join(self.dataroot, args.test_set))
        # tokenizer to process commment text
        # could set your path
        self.tokenizer = RobertaTokenizerFast.from_pretrained(
            pretrained_model_name_or_path="~/.cache/torch/hub/transformers/roberta-base")
        self.vocab_size = self.tokenizer.vocab_size # the length of vocabulary (text token)

        self.l_max_len = args.lang_seq_len # language length
        # self.a_max_len = args.audio_seq_len
        self.v_max_len = args.video_seq_len # video length

        self.pretrained_emb = None  # TODO # no LLM fine-tune??

    @staticmethod
    def read_json_file(path: str) -> Union[Dict, List[Dict],List]:
        import json
        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def read_npy_file(path: str) -> np.ndarray:
        return np.load(path, allow_pickle=True)

    @staticmethod
    def read_pkl_file(path: str) -> Dict:
        with open(path, 'rb') as f:
            return pickle.load(f)

    def get_video_feature(self, video_name): # get the raw video feature according to the filename in folder
        video_feat_path = os.path.join(self.video_feature_dir, f"{video_name}.npy")
        return self.read_npy_file(video_feat_path)
    
    def get_mask(self, length, maxlength):
        if length >= maxlength:
            mask = torch.ones((maxlength), dtype=torch.int64)
        else:
            mask = torch.zeros((maxlength), dtype=torch.int64)
            # mask[:length] = 1
            mask[:length]=torch.ones((length), dtype=torch.int64)
        return mask

    
    # transform str label to one hot tensor label
    def _encode_one_hot_label(self, opinion_label: str, emotion_label: str) -> Tuple[torch.Tensor, torch.Tensor]:
        max_opinion_idx = len(self.opinion_label_map)
        max_emotion_idx = len(self.emotion_label_map)

        opinion_vec = torch.zeros(max_opinion_idx)
        opinion_label_idx = self.opinion_label_map.get(opinion_label)
        opinion_vec[opinion_label_idx] = 1

        emotion_vec = torch.zeros(max_emotion_idx)
        emotion_label_ix = self.emotion_label_map.get(emotion_label)
        emotion_vec[emotion_label_ix] = 1

        return opinion_vec, emotion_vec
    # return comment text and the corresponding label tensor
    def get_video_current_comment_info(self, annotation: Dict) -> Dict:
        opinion_label = annotation.get("opinion_label")
        emotion_label = annotation.get("emotion_label")
        
        comment = annotation.get("comment")

        _opinion_label, _emotion_label = self._encode_one_hot_label(opinion_label, emotion_label)
        comment_info = {"emotion_label": _emotion_label, "opinion_label": _opinion_label, "comment": comment}

        return comment_info
    # sampling another comment for loss
    def get_video_other_comment_info(self, commentkey, videoid) -> Dict:
        # other_comment_ids = annotation.get("other_comment_ids")
        comments_in_video = self.video_to_comment[videoid]
        random_idx = random.choice(comments_in_video)
        while random_idx == commentkey:
            random_idx = random.choice(comments_in_video)

        # video_name = annotation.get("video_name")
        # select_comment = self.video_comment.get(video_name)[random_idx] # video_comment only for get another comment in same video
        # random.choice(comments_in_video)

        other_comment_info = self.get_video_current_comment_info(self.annotations.get(random_idx))
        return other_comment_info
    
    def __getitem__(self, idx):
        # annotation = self.annotations[idx]
        annotation_commentkey= self.label_data_select[idx]
        # video_feat = self.get_video_feature(annotation.get("video_name"))
        comment_label_data = self.annotations.get(annotation_commentkey)
        videoid = comment_label_data.get("video_file_id")
        video_feat = self.get_video_feature(videoid[:-4]) #exclude .mp4
        
        # comment_info: Dict = self.get_video_current_comment_info(annotation)
        comment_info: Dict = self.get_video_current_comment_info(comment_label_data)
        # other_comment_info: Dict = self.get_video_other_comment_info(annotation) # random sample another comment in each get 
        other_comment_info: Dict = self.get_video_other_comment_info(annotation_commentkey, videoid)
        # output
        output = dict()
        output = comment_label_data # {"video_file_id": "6919483488475286785.mp4",        "comment": "volatility = more opportunities",        "opinion_label": "positive",        "emotion_label": "trust",        "hashtag": "eth"    }
        output["comment_Key"] = annotation_commentkey
        output["video_feat"] = pad_feature(video_feat, self.v_max_len) # a tensor
        output["video_feat_rawlength"]=video_feat.shape[0]
        output["video_feat_mask"]=self.get_mask( video_feat.shape[0],self.v_max_len)
        output["comment_info"] = comment_info # {"emotion_label": _emotion_label, "opinion_label": _opinion_label, "comment": comment}
        output["other_comment_info"] = other_comment_info # same as comment_info

        return output

    def __len__(self):
        # return len(self.annotations)
        return len(self.label_data_select)
        # return 64

class CSMV_Dataset_r2plus1(Dataset):
    def __init__(self, name, args, token_to_ix=None, dataroot='data'):
        super(CSMV_Dataset_VideoMAEv2FPS16, self).__init__()
        assert name in ['train', 'valid', 'test', 'private']
        self.name = name
        self.args = args
        self.private_set = name == 'private'
        # self.dataroot = os.path.join(dataroot, 'mvc_data')
        self.dataroot = dataroot
        self.video_feature_dir = args.video_feature_dir

        self.emotion_label_map = self.read_json_file(os.path.join(self.dataroot, args.emotion_label_map)) #label mapping file of emotion
        self.opinion_label_map = self.read_json_file(os.path.join(self.dataroot, args.opinion_label_map)) #label mapping file of opinion
        # self.video_to_comment = self.read_pkl_file(os.path.join(self.dataroot, args.video_comment)) # input data, FOR resampling
        self.video_to_comment = self.read_json_file(os.path.join(self.dataroot, args.video_comment)) # input data, FOR resampling
        self.annotations = self.read_json_file(os.path.join(self.dataroot, args.annotations)) # annotation of input data, label data
        if name == 'train': # input data, get the commentkeys which use
            self.label_data_select = self.read_json_file(os.path.join(self.dataroot, args.train_set))
        elif name == 'valid':
            self.label_data_select = self.read_json_file(os.path.join(self.dataroot, args.dev_set))
        elif name == 'test':
            self.label_data_select = self.read_json_file(os.path.join(self.dataroot, args.test_set))
        # tokenizer to process commment text
        self.tokenizer = RobertaTokenizerFast.from_pretrained(
            pretrained_model_name_or_path="~/.cache/torch/hub/transformers/roberta-base")
        self.vocab_size = self.tokenizer.vocab_size # the length of vocabulary (text token)

        self.l_max_len = args.lang_seq_len # language length
        # self.a_max_len = args.audio_seq_len
        self.v_max_len = args.video_seq_len # video length

        self.pretrained_emb = None  # TODO # no LLM fine-tune??

    @staticmethod
    def read_json_file(path: str) -> Union[Dict, List[Dict],List]:
        import json
        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def read_npy_file(path: str) -> np.ndarray:
        return np.load(path, allow_pickle=True)

    @staticmethod
    def read_pkl_file(path: str) -> Dict:
        with open(path, 'rb') as f:
            return pickle.load(f)

    def get_video_feature(self, video_name): # get the raw video feature according to the filename in folder
        video_feat_path = os.path.join(self.video_feature_dir, f"{video_name}_r21d.npy")
        return self.read_npy_file(video_feat_path)
    
    # transform str label to one hot tensor label
    def _encode_one_hot_label(self, opinion_label: str, emotion_label: str) -> Tuple[torch.Tensor, torch.Tensor]:
        max_opinion_idx = len(self.opinion_label_map)
        max_emotion_idx = len(self.emotion_label_map)

        opinion_vec = torch.zeros(max_opinion_idx)
        opinion_label_idx = self.opinion_label_map.get(opinion_label)
        opinion_vec[opinion_label_idx] = 1

        emotion_vec = torch.zeros(max_emotion_idx)
        emotion_label_ix = self.emotion_label_map.get(emotion_label)
        emotion_vec[emotion_label_ix] = 1

        return opinion_vec, emotion_vec
    # return comment text and the corresponding label tensor
    def get_video_current_comment_info(self, annotation: Dict) -> Dict:
        opinion_label = annotation.get("opinion_label")
        emotion_label = annotation.get("emotion_label")
        
        comment = annotation.get("comment")

        _opinion_label, _emotion_label = self._encode_one_hot_label(opinion_label, emotion_label)
        comment_info = {"emotion_label": _emotion_label, "opinion_label": _opinion_label, "comment": comment}

        return comment_info
    # sampling another comment for loss
    def get_video_other_comment_info(self, commentkey, videoid) -> Dict:
        # other_comment_ids = annotation.get("other_comment_ids")
        comments_in_video = self.video_to_comment[videoid]
        random_idx = random.choice(comments_in_video)
        while random_idx == commentkey:
            random_idx = random.choice(comments_in_video)

        other_comment_info = self.get_video_current_comment_info(self.annotations.get(random_idx))
        return other_comment_info
    
    def __getitem__(self, idx):

        annotation_commentkey= self.label_data_select[idx]
        # video_feat = self.get_video_feature(annotation.get("video_name"))
        comment_label_data = self.annotations.get(annotation_commentkey)
        videoid = comment_label_data.get("video_file_id")
        video_feat = self.get_video_feature(videoid[:-4]) #exclude .mp4 in file name
        
        comment_info: Dict = self.get_video_current_comment_info(comment_label_data)

        other_comment_info: Dict = self.get_video_other_comment_info(annotation_commentkey, videoid)
        # output
        output = dict()
        output = comment_label_data # {"video_file_id": "6919483488475286785.mp4",        "comment": "volatility = more opportunities",        "opinion_label": "positive",        "emotion_label": "trust",        "hashtag": "eth"    }
        output["comment_Key"] = annotation_commentkey
        output["video_feat"] = pad_feature(video_feat, self.v_max_len) # a tensor
        output["comment_info"] = comment_info # {"emotion_label": _emotion_label, "opinion_label": _opinion_label, "comment": comment}
        output["other_comment_info"] = other_comment_info # same as comment_info

        return output

    def __len__(self):
        # return len(self.annotations)
        return len(self.label_data_select)
        # return 64
    

class CSMV_Dataset_VideoMAEv2FPS24(Dataset): #  fps 24
    def __init__(self, name, args, token_to_ix=None, dataroot='data'):
        super(CSMV_Dataset_VideoMAEv2FPS24, self).__init__()
        assert name in ['train', 'valid', 'test', 'private']
        self.name = name
        self.args = args
        self.private_set = name == 'private'
        self.dataroot = dataroot
        self.video_feature_dir = args.video_feature_dir

        self.emotion_label_map = self.read_json_file(os.path.join(self.dataroot, args.emotion_label_map)) #label mapping file of emotion
        self.opinion_label_map = self.read_json_file(os.path.join(self.dataroot, args.opinion_label_map)) #label mapping file of opinion
        # self.video_to_comment = self.read_pkl_file(os.path.join(self.dataroot, args.video_comment)) # input data, FOR resampling
        self.video_to_comment = self.read_json_file(os.path.join(self.dataroot, args.video_comment)) # input data, FOR resampling
        self.annotations = self.read_json_file(os.path.join(self.dataroot, args.annotations)) # annotation of input data, label data
        if name == 'train': # input data, get the commentkeys which use
            self.label_data_select = self.read_json_file(os.path.join(self.dataroot, args.train_set))
        elif name == 'valid':
            self.label_data_select = self.read_json_file(os.path.join(self.dataroot, args.dev_set))
        elif name == 'test':
            self.label_data_select = self.read_json_file(os.path.join(self.dataroot, args.test_set))
        # tokenizer to process commment text
        self.tokenizer = RobertaTokenizerFast.from_pretrained(
            pretrained_model_name_or_path="/home/jac/.cache/torch/hub/transformers/roberta-base")
        self.vocab_size = self.tokenizer.vocab_size # the length of vocabulary (text token)

        self.l_max_len = args.lang_seq_len # language length
        # self.a_max_len = args.audio_seq_len
        self.v_max_len = args.video_seq_len # video length

        self.pretrained_emb = None  # TODO # no LLM fine-tune??

    @staticmethod
    def read_json_file(path: str) -> Union[Dict, List[Dict],List]:
        import json
        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def read_npy_file(path: str) -> np.ndarray:
        return np.load(path, allow_pickle=True)

    @staticmethod
    def read_pkl_file(path: str) -> Dict:
        with open(path, 'rb') as f:
            return pickle.load(f)

    def get_video_feature(self, video_name): # get the raw video feature according to the filename in folder
        video_feat_path = os.path.join(self.video_feature_dir, f"{video_name}.npy")
        return self.read_npy_file(video_feat_path)
    
    # transform str label to one hot tensor label
    def _encode_one_hot_label(self, opinion_label: str, emotion_label: str) -> Tuple[torch.Tensor, torch.Tensor]:
        max_opinion_idx = len(self.opinion_label_map)
        max_emotion_idx = len(self.emotion_label_map)

        opinion_vec = torch.zeros(max_opinion_idx)
        opinion_label_idx = self.opinion_label_map.get(opinion_label)
        opinion_vec[opinion_label_idx] = 1

        emotion_vec = torch.zeros(max_emotion_idx)
        emotion_label_ix = self.emotion_label_map.get(emotion_label)
        emotion_vec[emotion_label_ix] = 1

        return opinion_vec, emotion_vec
    # return comment text and the corresponding label tensor
    def get_video_current_comment_info(self, annotation: Dict) -> Dict:
        opinion_label = annotation.get("opinion_label")
        emotion_label = annotation.get("emotion_label")
        
        comment = annotation.get("comment")

        _opinion_label, _emotion_label = self._encode_one_hot_label(opinion_label, emotion_label)
        comment_info = {"emotion_label": _emotion_label, "opinion_label": _opinion_label, "comment": comment}

        return comment_info
    # sampling another comment for loss
    def get_video_other_comment_info(self, commentkey, videoid) -> Dict:
        # other_comment_ids = annotation.get("other_comment_ids")
        comments_in_video = self.video_to_comment[videoid]
        random_idx = random.choice(comments_in_video)
        while random_idx == commentkey:
            random_idx = random.choice(comments_in_video)

        # video_name = annotation.get("video_name")
        # select_comment = self.video_comment.get(video_name)[random_idx] # video_comment only for get another comment in same video
        # random.choice(comments_in_video)

        other_comment_info = self.get_video_current_comment_info(self.annotations.get(random_idx))
        return other_comment_info
    
    def __getitem__(self, idx):
        # annotation = self.annotations[idx]
        annotation_commentkey= self.label_data_select[idx]
        # video_feat = self.get_video_feature(annotation.get("video_name"))
        comment_label_data = self.annotations.get(annotation_commentkey)
        videoid = comment_label_data.get("video_file_id")
        video_feat = self.get_video_feature(videoid[:-4]) #exclude .mp4
        
        # comment_info: Dict = self.get_video_current_comment_info(annotation)
        comment_info: Dict = self.get_video_current_comment_info(comment_label_data)
        # other_comment_info: Dict = self.get_video_other_comment_info(annotation) # random sample another comment in each get 
        other_comment_info: Dict = self.get_video_other_comment_info(annotation_commentkey, videoid)
        # output
        output = dict()
        output = comment_label_data # {"video_file_id": "6919483488475286785.mp4",        "comment": "volatility = more opportunities",        "opinion_label": "positive",        "emotion_label": "trust",        "hashtag": "eth"    }
        output["comment_Key"] = annotation_commentkey
        output["video_feat"] = pad_feature(video_feat, self.v_max_len) # a tensor
        output["comment_info"] = comment_info # {"emotion_label": _emotion_label, "opinion_label": _opinion_label, "comment": comment}
        output["other_comment_info"] = other_comment_info # same as comment_info

        return output

    def __len__(self):
        # return len(self.annotations)
        return len(self.label_data_select)