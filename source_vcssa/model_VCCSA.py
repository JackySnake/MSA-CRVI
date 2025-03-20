from audioop import mul
from distutils.command.config import config
import math
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.fc import MLP
from layers.layer_norm import LayerNorm
from transformers import RobertaModel, RobertaTokenizerFast
from easydict import EasyDict as eDict
from typing import Dict, List, Optional, Union, Tuple
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

# VisualEncoder config:visual_dim, visual_out_dim
# TextEncoder config:text_max_len
# Visual_CNN: config:visual_out_dim, visualcnn_dim
# Ground_weight config: visualcnn_dim, vtalign_dim, visual_length, text_dim
# Multiview_Attention config:text_dim, visualcnn_dim, vtalign_dim, fusion2text_dim
# MVCAnalysis_Model: args:mode,cuda | config: cnn_stack_num, fusion2text_dim, text_dim,head, opinion_num, emotion_num

# add dropout and padding dealing

class Conv1D(nn.Module): 
    def __init__(self, in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=True):
        super(Conv1D, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size, padding=padding,
                                stride=stride, bias=bias)

    def forward(self, x):
        # suppose all the input with shape (batch_size, seq_len, dim)
        x = x.transpose(1, 2)  # (batch_size, dim, seq_len)
        x = self.conv1d(x)
        return x.transpose(1, 2)  # (batch_size, seq_len, dim)

class VisualEncoder(nn.Module):# VisualEncoder config:visual_dim, visual_out_dim
    def __init__(self, configs: eDict):
        super(VisualEncoder, self).__init__()
        self.linear = Conv1D(in_dim=configs.visual_dim, 
                             out_dim=configs.visual_out_dim,
                             kernel_size=1,
                             stride=1,
                             bias=True,
                             padding=0)

    def forward(self, visual_features):
        # the input visual feature with shape (batch_size, seq_len, visual_dim)

        output = self.linear(visual_features)  # (batch_size, seq_len, dim)
        return output

class TextEncoder(nn.Module):# TextEncoder config:text_max_len
    def __init__(self, pre_trained_LM_path, configs: eDict):
        super(TextEncoder, self).__init__()
        self.text_max_len = configs.text_len  # max language text length
        self.tokenizer = RobertaTokenizerFast.from_pretrained(
            pretrained_model_name_or_path = pre_trained_LM_path)
        self.text_encoder = RobertaModel.from_pretrained(
            pretrained_model_name_or_path = pre_trained_LM_path) 

    def forward(self, comment: Dict):
        tokens = self.tokenizer.batch_encode_plus(comment.get("comment"),
                                                  padding="max_length",
                                                  return_tensors="pt",
                                                  max_length=self.text_max_len,
                                                  truncation=True)
        tokens = tokens.to(self.text_encoder.device)
        tokens_encoder = self.text_encoder(**tokens)

        text_encoder_globle = tokens_encoder.pooler_output
        text_encoder_token = tokens_encoder[0]

        # text_mask = tokens.attention_mask.ne(1).bool()
        text_mask = tokens.attention_mask
        text_pos = torch.zeros_like(text_encoder_globle)

        output = {"globle_feature": text_encoder_globle,
                  "token_feature": text_encoder_token,
                  "original_fearture": tokens_encoder,
                  "position": text_pos,
                  "mask": text_mask,
                  "tokens": tokens
                  }
        return output

class Visual_Temporal_CNN(nn.Module):# Visual_Temporal_CNN: config:visual_out_dim, visualcnn_dim
    def __init__(self, cfg):
        super(Visual_Temporal_CNN, self).__init__()
        self.conv = nn.Conv1d(in_channels=cfg.visual_out_dim, out_channels=cfg.visualcnn_dim, \
                kernel_size=3, stride=1, bias=True, padding=1               
                )# need padding
        self.acticate_f = nn.ReLU()

    def forward(self, x):
        # suppose all the input with shape (batch_size, seq_len, dim)
        x = x.transpose(1, 2)  # (batch_size, dim, seq_len)
        x = self.conv(x)
        x = x.transpose(1, 2) # (batch_size, seq_len, dim)
        x = self.acticate_f(x)
        return x  

class Multiview_Attention(nn.Module): #Multiview_Attention config:text_dim, visualcnn_dim, vtalign_dim, fusion2text_dim
    def __init__(self, cfg):
        super(Multiview_Attention, self).__init__()
        self.ffn_text = nn.Linear(cfg.text_dim, cfg.vtalign_ground_dim)
        self.ffn_visual = nn.Linear(cfg.visualcnn_dim, cfg.vtalign_ground_dim)
        self.att_softmax = nn.Softmax(dim=-1)

    def forward(self, text_feature, multiview_grounding_feature):
        text_feat = self.ffn_text(text_feature)
        multiview_feat = self.ffn_visual(multiview_grounding_feature)

        attention_matrix = torch.einsum("blf,bvf->blv",[text_feat, multiview_feat])
        attention_matrix = self.att_softmax(attention_matrix)

        multiview_fusion = torch.einsum("blv,bvf->blf",attention_matrix, multiview_grounding_feature)

        return multiview_fusion, attention_matrix
        
class Consensus_Transformer_inview(nn.Module): #dropout
    def __init__(self, cfg, consensus_mask=True):
        super(Consensus_Transformer_inview, self).__init__()
        # for liner trans

        self.visual_dim = cfg.visualcnn_dim
        self.text_dim = cfg.text_dim
        self.input_dim = cfg.consensus_Transformer_fdim

        self.head = cfg.consensus_Transformer_head
        self.consensus_mask = consensus_mask

        self.visualLinear = nn.Linear(self.visual_dim, self.input_dim)
        self.textLinear = nn.Linear(self.text_dim, self.input_dim)
        self.acticate_f = nn.ReLU()
        self.transformer = nn.MultiheadAttention(embed_dim=self.input_dim, num_heads=self.head, dropout=cfg.dropout, batch_first=True)

        
    def attn_mask_matrix_cal(self, batch_size, visual_max_length, text_max_length):
        # attention mask matrix

        attn_mask_video = torch.rand(1,1,visual_max_length+text_max_length+1, requires_grad=False).bool()

        attn_mask_video[:,:,:] = False
        attn_mask_video[:,:,visual_max_length+1:] = True
        attn_mask_video = attn_mask_video.repeat(batch_size*self.head, visual_max_length, 1)

        attn_mask_text = torch.rand(1,1,visual_max_length+text_max_length+1, requires_grad=False).bool()
        attn_mask_text[:,:,:] = False
        attn_mask_text[:,:,0:visual_max_length] = True
        attn_mask_text = attn_mask_text.repeat(batch_size*self.head, text_max_length, 1)

        attn_mask_consensus  = torch.rand(1,1,visual_max_length+text_max_length+1, requires_grad=False).bool()
        attn_mask_consensus[:,:,:] = False
        attn_mask_consensus = attn_mask_consensus.repeat(batch_size*self.head,1,1)

        attn_mask = torch.cat((attn_mask_video, attn_mask_consensus, attn_mask_text), dim = 1)

        return attn_mask

    def key_padding_mask_matrix_cal(self, batch_size, visual_max_length, text_max_length, visual_mask, text_mask): #return (batchsize, whole_length=vl+1+tl)
        mask_tensor = torch.rand(batch_size,visual_max_length+text_max_length+1, requires_grad=False).bool()
        mask_tensor[:,:] = False
        mask_tensor[:,:visual_max_length] = visual_mask.ne(1).bool()
        mask_tensor[:,visual_max_length+1:] = text_mask.ne(1).bool()
        return mask_tensor

   
    def forward(self, visual_input, text_input, consensus_token, visual_mask, text_mask): #consensus_token:[B,1,consensus_Transformer_fdim]
        #Actually, the three value should get from cfg
        batch_size = visual_input.shape[0]
        visual_length = visual_input.shape[1]
        text_length = text_input.shape[1]
        
        visual_trans_input = self.visualLinear(visual_input)
        visual_trans_input = self.acticate_f(visual_trans_input)
        text_trans_input = self.textLinear(text_input)
        text_trans_input = self.acticate_f(text_trans_input)


        consensus_input = consensus_token

        transformer_input = torch.cat((visual_trans_input,consensus_input,text_trans_input),dim=1) 
        padding_mask = self.key_padding_mask_matrix_cal(batch_size, visual_length, text_length, visual_mask, text_mask)
        padding_mask = padding_mask.to(transformer_input.device)

        if self.consensus_mask == True:
            attn_mask = self.attn_mask_matrix_cal(batch_size, visual_length, text_length)
            attn_mask = attn_mask.to(transformer_input.device)
            trans_output, attn_score = self.transformer(query=transformer_input, 
                                                        key=transformer_input, 
                                                        value=transformer_input, 
                                                        key_padding_mask=padding_mask,
                                                        attn_mask = attn_mask, average_attn_weights=True)
        else:
            print("self.consensus_mask == False. So no Mask consensus.")
            trans_output, attn_score = self.transformer(query=transformer_input, 
                                            key=transformer_input, 
                                            value=transformer_input,
                                            key_padding_mask=padding_mask, 
                                            attn_mask = None, average_attn_weights=True)

        return trans_output, attn_score
       

class Classifier_header(nn.Module):
    def __init__(self, input_dim, label_num):
        super(Classifier_header, self).__init__()
        self.input_dim = input_dim
        self.label_num = label_num
        
        self.classLinear = nn.Linear(self.input_dim, self.label_num)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, input_feature):
        feat = self.classLinear(input_feature)
        preds = self.softmax(feat)
        return preds

class MVCAnalysis_Model(nn.Module):  # MVCAnalysis_Model: args:cuda | config:  cnn_stack_num, fusion2text_dim, text_dim,head, opinion_num, emotion_num
    def __init__(self, args, cfg: eDict): 
        super(MVCAnalysis_Model, self).__init__()

        self.cfg = cfg
        # self.mode = args.mode #train, dev, test
        self.pretrained_model = args.pre_trained_LM

        self.visual_encoder = VisualEncoder(cfg)
        self.text_encoder = TextEncoder(self.pretrained_model, cfg)
        
        self.stack_cnn = [Visual_Temporal_CNN(cfg) for i in range(cfg.multiview_num)]
        self.stack_cnn = nn.ModuleList(self.stack_cnn)

        self.consensus_token = nn.Parameter(torch.rand((1,cfg.consensus_Transformer_fdim), requires_grad = True)) 

        self.consensus_mask = cfg.consensus_mask
        self.stack_Consensus_Transformer_inview = [Consensus_Transformer_inview(cfg, consensus_mask=self.consensus_mask) for i in range(cfg.multiview_num)]
        self.stack_Consensus_Transformer_inview = nn.ModuleList(self.stack_Consensus_Transformer_inview)

        self.grounding_Attention = nn.MultiheadAttention(embed_dim=cfg.consensus_Transformer_fdim, 
                                    num_heads=cfg.groundAttention_head, 
                                    dropout=cfg.dropout,
                                    kdim=cfg.visualcnn_dim, 
                                    vdim=cfg.visualcnn_dim,
                                    batch_first= True)

        self.ground_lstm = nn.LSTM(input_size = cfg.groundAttention_head, 
                                    hidden_size=cfg.visual_length, 
                                    batch_first = True)
        self.ground_weight_filter = nn.ReLU()
        
        self.text_transformer = nn.MultiheadAttention(embed_dim = cfg.text_dim, 
                                                        num_heads=cfg.final_Transformer_head, 
                                                        dropout= cfg.dropout,batch_first=True)

        self.multiview_attention_fusion = Multiview_Attention(cfg)

        self.dropout = nn.Dropout(cfg.dropout)

        self.fusion_linear = nn.Linear(cfg.visualcnn_dim + cfg.text_dim, cfg.text_dim)
        self.activate_f = nn.ReLU()
        
        self.opinion_classifier_head = Classifier_header(cfg.text_dim, cfg.opinion_num)
        self.emotion_classifier_head = Classifier_header(cfg.text_dim, cfg.emotion_num)

        self.opinion_loss = Focal_Loss(labelnum=cfg.opinion_num)
        self.emotion_loss = Focal_Loss(labelnum=cfg.emotion_num)

    def forward(self, batch): #dropout
        
        video_raw_feat = batch.get('video_feat')
        video_raw_feat = video_raw_feat.to(self.visual_encoder.linear.conv1d.weight.device)
        video_raw_feat = self.visual_encoder(video_raw_feat)
        video_raw_mask = batch.get('video_feat_mask')
        comment_info = batch.get("comment_info")
        text_raw_feat= self.text_encoder(comment_info)

        text_token_feat = text_raw_feat.get("token_feature")
        text_token_mask = text_raw_feat.get("mask")

        (final_feature_To_predict,
         _, _, 
        consensun_attention_multiview, 
        consensus_gate_multiview, 
        text_change_multiview,
        multiview_attention)  = self.forward_feature(video_raw_feat, text_token_feat, video_raw_mask, text_token_mask)

        final_feature_To_predict = self.dropout(final_feature_To_predict)

        opinion_preds = self.opinion_classifier_head(final_feature_To_predict)
        emotion_preds = self.emotion_classifier_head(final_feature_To_predict)

        output = {"opinion_predict": opinion_preds,
                  "emotion_predict": emotion_preds
                  }

        if self.training:
            opinion_loss = self.opinion_loss(opinion_preds, \
                comment_info['opinion_label'].to(opinion_preds.device))
            emotion_loss = self.emotion_loss(emotion_preds, \
                comment_info['emotion_label'].to(emotion_preds.device))

            output["opinion_loss"] = opinion_loss
            output["emotion_loss"] = emotion_loss

        return output

    def forward_feature(self, visual_fearture, text_feature, visual_mask, text_mask):
        
        batch_size = visual_fearture.shape[0]
        consensus_feature = self.consensus_token.repeat(batch_size, 1, 1)
        
        cnn_stack_feat_multiview = []
        consensus_transformer_multiview = []
        consensus_transformer_score_multiview = []
        consensus_feature_multiview = []
        consensus_grounding_multiview = []
        consensus_attention_multiview = []
        text_change_multiview = []
        consensus_gate_multiview = []        
        

        for i in range(self.cfg.multiview_num): # 
            if i == 0:
                cnn_feature = self.stack_cnn[i](visual_fearture)
                cnn_stack_feat_multiview.append(cnn_feature)

                cons_trans_feature, attn_socre = self.stack_Consensus_Transformer_inview[i](cnn_feature,
                                                                                         text_feature,
                                                                                         consensus_feature
                                                                                         , visual_mask, text_mask)
                consensus_transformer_multiview.append(cons_trans_feature)
                consensus_transformer_score_multiview.append(attn_socre)
                consensus_feat = cons_trans_feature[:,visual_fearture.shape[1],:]
                consensus_feat = consensus_feat.reshape(consensus_feat.shape[0], 1, consensus_feat.shape[-1])
                consensus_feature_multiview.append(consensus_feat)

                _, attention_clue  = self.grounding_Attention(query=consensus_feat, key=cnn_feature, value=cnn_feature, 
                                                                average_attn_weights=False)
                attention_clue = attention_clue.reshape(attention_clue.shape[0],attention_clue.shape[1],attention_clue.shape[-1])
                attention_clue = attention_clue.transpose(1,2)
                _,(_, c_out) = self.ground_lstm(attention_clue)
                cg_score = c_out.reshape(c_out.shape[1], c_out.shape[2])
                cg_score = self.ground_weight_filter(cg_score)

                consensus_grounding_feature = torch.einsum("bv,bvf->bf",[cg_score, cnn_feature])
                consensus_grounding_feature = consensus_grounding_feature.unsqueeze(dim = 1)

                consensus_grounding_multiview.append(consensus_grounding_feature)
                consensus_attention_multiview.append(cg_score)

            else:
                cnn_feature = self.stack_cnn[i](cnn_stack_feat_multiview[-1])
                cnn_stack_feat_multiview.append(cnn_feature)

                cons_trans_feature, attn_socre = self.stack_Consensus_Transformer_inview[i](cnn_feature,
                                                                                         text_feature,
                                                                                         consensus_feature
                                                                                         , visual_mask, text_mask)
                consensus_transformer_multiview.append(cons_trans_feature)
                consensus_transformer_score_multiview.append(attn_socre)
                consensus_feat = cons_trans_feature[:,visual_fearture.shape[1],:]
                consensus_feat = consensus_feat.reshape(consensus_feat.shape[0], 1, consensus_feat.shape[-1])
                consensus_feature_multiview.append(consensus_feat)

                _, attention_clue  = self.grounding_Attention(query=consensus_feat, key=cnn_feature, value=cnn_feature, 
                                                                average_attn_weights=False)
                attention_clue = attention_clue.reshape(attention_clue.shape[0],attention_clue.shape[1],attention_clue.shape[-1])
                attention_clue = attention_clue.transpose(1,2)
                _,(_, c_out) = self.ground_lstm(attention_clue)
                cg_score = c_out.reshape(c_out.shape[1], c_out.shape[2])
                cg_score = self.ground_weight_filter(cg_score)

                consensus_grounding_feature = torch.einsum("bv,bvf->bf",[cg_score, cnn_feature])
                consensus_grounding_feature = consensus_grounding_feature.unsqueeze(dim = 1)

                consensus_grounding_multiview.append(consensus_grounding_feature)
                consensus_attention_multiview.append(cg_score)

        multiview_grounding_feature = torch.cat(consensus_grounding_multiview, dim=1)
        multiview_grounding_feature, multiview_attn =self.multiview_attention_fusion(text_feature, multiview_grounding_feature)

        multiview_grounding_feature = torch.mul(text_mask.unsqueeze(2), multiview_grounding_feature) 

        text_cat = torch.cat((text_feature, multiview_grounding_feature), dim=-1)
        text_fusion_pre = self.fusion_linear(text_cat)
        text_fusion_pre = self.activate_f(text_fusion_pre)

        text_fusion,  attn_output_weight = self.text_transformer(query=text_fusion_pre, 
                                                        key=text_fusion_pre, 
                                                        value=text_fusion_pre,
                                                        key_padding_mask=text_mask.ne(1).bool()) 
        final_feature = torch.max_pool2d(text_fusion,kernel_size=(text_fusion.shape[1],1))
        final_feature = final_feature.reshape(final_feature.shape[0], final_feature.shape[-1])
        
        return (final_feature, 
                consensus_transformer_score_multiview,
                consensus_feature_multiview,
                consensus_attention_multiview,
                consensus_gate_multiview,
                text_change_multiview,
                multiview_attn)


class Focal_Loss(nn.Module):
    def __init__(self, weight=None, gamma=2, labelnum=2):
        super(Focal_Loss,self).__init__()
        self.gamma = gamma
        if weight != None:
            self.weight = weight        
        else:
            avg_weight = 1/labelnum
            wlist = []
            for i in range(labelnum):
                wlist.append(avg_weight)

            self.weight = torch.tensor(wlist)
        self.weight = nn.Parameter(self.weight, requires_grad = False)
    def forward(self, preds, labels):
        """
        preds:logist输出值
        labels:标签
        """
        eps = 1e-7 # ?

        ce = ((-1 * torch.log(preds+eps))+eps) * labels
        # print(ce)
        floss = torch.pow((1-preds), self.gamma) * ce
        # print(floss)
        floss = torch.mul(floss, self.weight)
        # print(floss)
        floss = torch.sum(floss, dim=1)
        # print(floss)
        floss = torch.mean(floss)
        if floss.item() < 0:
            print("**********************")
            print("floss < 0\n")
            print("preds:")
            print(preds)
            print("labels:")
            print(labels)
            print("**********************")
        return  floss# average
