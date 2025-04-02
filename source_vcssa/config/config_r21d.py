# VisualEncoder config:visual_dim, visual_out_dim
# TextEncoder config:text_max_len
# Visual_CNN: config:visual_out_dim, visualcnn_dim
# Ground_weight config: visualcnn_dim, vtalign_dim, visual_length, text_dim
# Multiview_Attention config:text_dim, visualcnn_dim, vtalign_dim, fusion2text_dim
# MVCAnalysis_Model: args:mode,cuda | config: cnn_stack_num, fusion2text_dim, text_dim,head, opinion_num, emotion_num
# dropout

model_cfg = dict(
    text_len = 512,
    visual_length = 180, # 
    visual_dim = 512,
    visual_out_dim = 768,
    visualcnn_dim = 768,
    vtalign_ground_dim = 768,
    text_dim = 768,
    multiview_num = 4,
    consensus_Transformer_head = 8,
    consensus_Transformer_fdim = 512,
    final_Transformer_head = 4,
    opinion_num = 3,
    emotion_num = 8,
    dropout = 0.2,
    margin_loss_delta = 0.5,
    gate_token_dim = 256, 
    consensus_mask = True,
    groundAttention_head = 16,
    transformer_layer = 2
)
