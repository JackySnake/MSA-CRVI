### Parameter for use:
# output_dir: evaluation result output
# video_feature: video feature path of CSMV
# comment_text: comment text path of CSMV, including various mapping json file(opinion_label_map etc.)
# model_dir: the path of checkpoint. The main_eval.py will select the best performace in val set to evalate the performace of test set.

#note: the path is relatetive path, the absolute path is also good

output_dir=../output/eval
video_feature=../dataset/csmv/visual-feature
comment_text=../dataset/csmv/commentDataset
model_dir=../ckpt/vccsa-3407
batch_size=128
num_workers=8
# eval_name=VCCSA-eval
gpuid=0


echo ${train_name} start running...

python ../main_eval.py \
--dataset CSMV \
--lang_seq_len 512 \
--video_seq_len 180 \
--batch_size ${batch_size}  \
--num_workers ${num_workers} \
--output ${output_dir} \
--video_feature_dir ${video_feature} \
--datadir ${comment_text} \
--opinion_label_map opinion_label_map.json \
--emotion_label_map emotion_label_map.json \
--video_comment video_to_comment.json \
--annotations lable_data_dict.json \
--train_set train_set.json \
--dev_set dev_set.json \
--test_set test_set.json \
--model_dir ${model_dir} \
--model_name VCCSA \
--cuda ${gpuid} \
--name test
