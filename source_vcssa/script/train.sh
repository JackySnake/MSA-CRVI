### Parameter for use:
# lr_base: learning rate
# seed
# output_dir:train output
# train_name: determing the checkpoint save directory path in output_dir
# video_feature: video feature path of CSMV
# comment_text: comment text path of CSMV, including various mapping json file(opinion_label_map etc.)

#note: the path is relatetive path, the absolute path is also good

output_dir=../ckpt
video_feature=../dataset/csmv/visual-feature
comment_text=../dataset/csmv/commentDataset
batch_size=16
num_workers=8
seed=3407
train_name=VCCSA-seed${seed}
lr_base=0.00005
gpuid=0

echo ${train_name} start running...

python ../main.py \
--dataset CSMV \
--batch_size ${batch_size} \
--num_workers ${num_workers} \
--seed ${seed} \
--lang_seq_len 512 \
--video_seq_len 180 \
--optim adam \
--lr_base ${lr_base} \
--dropout_r 0.5 \
--output ${output_dir} \
--video_feature_dir ${video_feature_dir} \ 
--datadir ${comment_text} \
--opinion_label_map opinion_label_map.json \
--emotion_label_map emotion_label_map.json \
--video_comment video_to_comment.json \
--annotations lable_data_dict.json \
--train_set train_set.json \
--dev_set dev_set.json \
--test_set test_set.json \
--name ${train_name} \
--cuda ${gpuid}
