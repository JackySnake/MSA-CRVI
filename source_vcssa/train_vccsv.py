from asyncio import open_connection
import pickle
from telnetlib import STATUS
from unittest import result
import tensorboard
import torch
import torch.nn as nn
import time
import numpy as np
import os
from utils.pred_func import *
from typing import Dict
from sklearn import metrics
from typing import Dict, Optional, List, Tuple, Union

from tensorboardX import SummaryWriter

from loguru import logger
import json

from typing import Dict, Optional, List, Tuple, Union
from tqdm import tqdm


def read_json_file(path: str) -> Union[Dict, List[Dict], List]:
    import json
    with open(path, "r") as f:
        return json.load(f)


def write_json_file(file_name, data) -> None:
    logger.info(f"Writing {file_name}")
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)


def write_pkl_file(file_name, data) -> None:
    logger.info(f"Writing {file_name}")
    with open(file_name, "wb") as f:
        pickle.dump(data, f)


def train(net, train_loader, eval_loader, optim, args, scheduler):
    logfile = open(args.output + "/" + args.name + '/log_run.txt', 'w+')
    logfile.write(str(args))
    writer = SummaryWriter(args.output + "/" + args.name + '/train_tersorboard')
    # initialize the loss and other optimizer parameter
    # loss_sum = 0
    best_eval_accuracy = 0.0
    early_stop = 0
    decay_count = 0
    best_epoch = 0

    tensorboard_steps = 0

    emotion_label_map = read_json_file(os.path.join(args.datadir,
                                                    args.emotion_label_map))  #label mapping file of emotion
    opinion_label_map = read_json_file(os.path.join(args.datadir,
                                                    args.opinion_label_map))  #label mapping file of opinion

    eval_accuracies = []  # to record evaluate accuracy
    for epoch in range(0, args.max_epoch):  # epoch loop

        epoch_loss, epoch_op_loss, epoch_em_loss, epoch_mag_loss = 0, 0, 0, 0
        op_aux_loss_sum = 0
        emo_aux_loss_sum = 0

        time_start = time.time()  # record train time

        for step, batch_data in enumerate(train_loader):  # step : a "backward"

            optim.zero_grad()  # reset the grad to 0
            output: Dict = net(batch_data)  # forward
            op_loss = output.get("opinion_loss")
            emo_loss = output.get("emotion_loss")

            # if args.aux_task == True:
            #     op_aux_loss = output.get("opinion_aux_loss")
            #     emo_aux_loss = output.get("emotion_aux_loss")
            #     loss = op_loss + emo_loss + op_aux_loss + emo_aux_loss
            # else:
            #     loss = op_loss + emo_loss
                # loss = op_loss + emo_loss + margin_loss
            loss = op_loss + emo_loss

            loss.backward()  # backward

            # Gradient norm clipping
            if args.grad_norm_clip > 0:
                nn.utils.clip_grad_norm_(net.parameters(), args.grad_norm_clip)

            optim.step()
            scheduler.step()

            epoch_loss += loss
            epoch_op_loss += output.get("opinion_loss")
            epoch_em_loss += output.get("emotion_loss")
            # epoch_mag_loss += output.get("margin_loss")
            # if args.aux_task == True:
            #     op_aux_loss_sum += op_aux_loss
            #     emo_aux_loss_sum += emo_aux_loss

            if (step + 1) % 50 == 0:
                writer.add_scalar("loss", loss.item(), tensorboard_steps)
                writer.add_scalar("opinion_loss", output.get("opinion_loss").item(), tensorboard_steps)
                writer.add_scalar("emotion_loss", output.get("emotion_loss").item(), tensorboard_steps)
                # if args.aux_task == True:
                #     writer.add_scalar("opinion_aux_loss", output.get("opinion_aux_loss").item(), tensorboard_steps)
                #     writer.add_scalar("emotion_aux_loss", output.get("emotion_aux_loss").item(), tensorboard_steps)

                tensorboard_steps += 1
            if args.aux_task == True:
                print(
                    "\r[Epoch %2d][Step %4d/%4d] Loss_sum: %.4f,opinion_loss: %.4f, emotion_loss: %.4f, \
                        opinion_aux_loss: %.4f, emotion_aux_loss: %.4f, Lr: %.2e, %4d m "
                    "remaining" % (
                        epoch + 1,
                        step,
                        int(len(train_loader.dataset) / args.batch_size),
                        loss,
                        output.get("opinion_loss"),
                        output.get("emotion_loss"),
                        output.get("opinion_aux_loss"),
                        output.get("emotion_aux_loss"),
                        # *[group['lr'] for group in optim.param_groups],
                        [group['lr'] for group in optim.param_groups][0],
                        ((time.time() - time_start) / (step + 1)) *
                        ((len(train_loader.dataset) / args.batch_size) - step) / 60,
                    ),
                    end='          ')
            else:
                print(
                    "\r[Epoch %2d][Step %4d/%4d] Loss_sum: %.4f,opinion_loss: %.4f, emotion_loss: %.4f, Lr: %.2e, %4d m"
                    "remaining" % (
                        epoch + 1,
                        step,
                        int(len(train_loader.dataset) / args.batch_size),
                        loss,
                        output.get('opinion_loss'),
                        output.get('emotion_loss'),
                        # *[group['lr'] for group in optim.param_groups],
                        [group['lr'] for group in optim.param_groups][0],
                        ((time.time() - time_start) / (step + 1)) *
                        ((len(train_loader.dataset) / args.batch_size) - step) / 60,
                    ),
                    end='          ')

        time_end = time.time()
        elapse_time = time_end - time_start
        print('Finished in {}s'.format(int(elapse_time)))
        if args.aux_task == True:
            training_epoch_status ="\r[Epoch %2d]: Loss_sum: %.4f,opinion_loss: %.4f, emotion_loss: %.4f, \
                opinion_aux_loss: %.4f, emotion_aux_loss: %.4f" \
                    % (
                            epoch + 1,
                            epoch_loss,
                            epoch_op_loss,
                            epoch_em_loss,
                            op_aux_loss_sum,
                            emo_aux_loss_sum
                    )
        else:
            training_epoch_status ="\r[Epoch %2d]: Loss_sum: %.4f,opinion_loss: %.4f, emotion_loss: %.4f" \
                    % (
                            epoch + 1,
                            epoch_loss,
                            epoch_op_loss,
                            epoch_em_loss
                    )
        print(training_epoch_status)
        logfile.write(training_epoch_status)
        # tensorboard
        writer.add_scalar("epoch_loss", epoch_loss.item(), (epoch + 1))
        writer.add_scalar("epoch_opinion_loss", epoch_op_loss.item(), (epoch + 1))
        writer.add_scalar("epoch_emotion_loss", epoch_em_loss.item(), (epoch + 1))
        # writer.add_scalar("epoch_margin_loss", epoch_mag_loss.item(), (epoch+1))
        # if args.aux_task == True:
        #     writer.add_scalar("epoch_opinion_aux_loss", op_aux_loss_sum.item(), (epoch + 1))
        #     writer.add_scalar("epoch_emotion_aux_loss", emo_aux_loss_sum.item(), (epoch + 1))

        epoch_finish = epoch + 1
        if args.aux_task == True:
            write_json_file(args.output + "/" + args.name +'/'+'loss_epoc_'+str(epoch_finish)+'.json', \
                {"epoch_loss": epoch_loss.item(), \
                    "epoch_opinion_loss": epoch_op_loss.item(), \
                        "epoch_emotion_loss": epoch_em_loss.item(), \
                            "epoch_opinion_aux_loss": op_aux_loss_sum.item(), \
                                "epoch_emotion_aux_loss": emo_aux_loss_sum.item()
                        })
        else:
            write_json_file(args.output + "/" + args.name +'/'+'loss_epoc_'+str(epoch_finish)+'.json', \
                {"epoch_loss": epoch_loss.item(), \
                    "epoch_opinion_loss": epoch_op_loss.item(), \
                        "epoch_emotion_loss": epoch_em_loss.item()   \
                            # , "epoch_margin_loss": epoch_mag_loss.item()


                        })

        # Logging
        if args.aux_task == True:

            logfile.write('Epoch: ' + str(epoch_finish) + ', Loss: ' + str(epoch_loss / len(train_loader.dataset)) +
                          ', opinion_loss' + str(epoch_op_loss) + ', emotion_loss' + str(epoch_em_loss) +
                          ', opinion_aux_loss' + str(op_aux_loss_sum) + ', emotion_aux_loss' + str(emo_aux_loss_sum) +
                          ', Lr: ' + str([group['lr'] for group in optim.param_groups]) + '\n' + 'Elapsed time: ' +
                          str(int(elapse_time)) + ', Speed(s/batch): ' + str(elapse_time / step) + '\n\n')
        else:
            logfile.write('Epoch: ' + str(epoch_finish) + ', Loss: ' + str(epoch_loss / len(train_loader.dataset)) +
                          ', opinion_loss' + str(epoch_op_loss) + ', emotion_loss' + str(epoch_em_loss) +
                          # ', margin_loss' + str(epoch_mag_loss) +
                          ', Lr: ' + str([group['lr'] for group in optim.param_groups]) + '\n' + 'Elapsed time: ' +
                          str(int(elapse_time)) + ', Speed(s/batch): ' + str(elapse_time / step) + '\n\n')
        epoch_loss, epoch_op_loss, epoch_em_loss, epoch_mag_loss = 0, 0, 0, 0
        op_aux_loss_sum, emo_aux_loss_sum = 0, 0
        # Eval
        if epoch_finish >= args.eval_start:
            print('Evaluation...')
            net.eval()
            accuracy, result = evaluate(net, eval_loader, args)
            net.train()
            print('Accuracy :' + str(accuracy))
            eval_accuracies.append(accuracy)

            write_pkl_file(args.output + "/" + args.name + '/' + "dev_predict_" + str(epoch_finish) + ".pkl", result)
            write_json_file(args.output + "/" + args.name + '/' + "dev_performance_" + str(epoch_finish) + ".json",
                            accuracy)

            op_mi_precision = accuracy.get("opinion").get("micro").get("precision")
            em_mi_precision = accuracy.get("emotion").get("micro").get("precision")
            op_mi_recall = accuracy.get("opinion").get("micro").get("recal")
            em_mi_recall = accuracy.get("emotion").get("micro").get("recal")
            opinion_performance = accuracy.get("opinion").get("micro").get("f1_score")  # opinion f1
            emotion_performance = accuracy.get("emotion").get("micro").get("f1_score")  # emotino f1

            writer.add_scalar("opinion_micro_precision", op_mi_precision, (epoch + 1))
            writer.add_scalar("emotion_micro_precision", em_mi_precision, (epoch + 1))
            writer.add_scalar("opinion_micro_recall", op_mi_recall, (epoch + 1))
            writer.add_scalar("emotion_micro_recall", em_mi_recall, (epoch + 1))
            writer.add_scalar("opinion_micro_F1", opinion_performance, (epoch + 1))
            writer.add_scalar("emotion_micro_F1", emotion_performance, (epoch + 1))

            op_ma_precision = accuracy.get("opinion").get("micro").get("precision")
            em_ma_precision = accuracy.get("emotion").get("micro").get("precision")
            op_ma_recall = accuracy.get("opinion").get("micro").get("recal")
            em_ma_recall = accuracy.get("emotion").get("micro").get("recal")
            opinion_performance_ma = accuracy.get("opinion").get("micro").get("f1_score")  # opinion f1
            emotion_performance_ma = accuracy.get("emotion").get("micro").get("f1_score")  # emotino f1

            writer.add_scalar("opinion_macro_precision", op_ma_precision, (epoch + 1))
            writer.add_scalar("emotion_macro_precision", em_ma_precision, (epoch + 1))
            writer.add_scalar("opinion_macro_recall", op_ma_recall, (epoch + 1))
            writer.add_scalar("emotion_macro_recall", em_ma_recall, (epoch + 1))
            writer.add_scalar("opinion_macro_F1", opinion_performance_ma, (epoch + 1))
            writer.add_scalar("emotion_macro_F1", emotion_performance_ma, (epoch + 1))

            performace = opinion_performance + emotion_performance  
            logfile.write('Epoch ' + str(epoch_finish) + ' Accuracy :' + str(accuracy))  # record the performance
            state = {
                'state_dict': net.state_dict(),
                'optimizer': optim.state_dict(),
                'args': args,
            }
            torch.save(
                state, args.output + "/" + args.name + '/best' + str(args.seed) + '_' + str(performace) + '_' +
                str(epoch_finish) + '.pkl')
            if performace > best_eval_accuracy:
                best_eval_accuracy = performace
                best_epoch = epoch_finish
                print("best eopch: " + str(best_epoch) + " , performance: " + str(best_eval_accuracy))


@torch.no_grad()
def evaluate(net,
             eval_loader,
             args,
             opinion_label_num=3,
             opinion_label_map=None,
             emotion_label_num=8,
             emotion_label_map=None):
    accuracy = {}
    # net_STATUS = net.training
    # net.train(False)
    # net.eval()
    opinion_labels_list = list(range(opinion_label_num))
    emotion_labels_list = list(range(emotion_label_num))
    comments_key = []  # dataid
    opinion_preds = []  # record the prediction results
    opinion_preds_classidx = []  # record the prediction results on class
    emotion_preds = []  # record the prediction results
    emotion_preds_classidx = []  # record the prediction results on class
    multiview_attention = []  # record the multiview attention score

    opinions_label = []  # ground truth tensor
    emotions_label = []  # ground truth tensor
    opinions_label_classindex = []  # ground truth class
    emotions_label_classindex = []  # ground truth class
    with torch.no_grad():
        for step, batch_data in tqdm(enumerate(eval_loader)):
            # gruond truth and data keys
            comments_key.extend(batch_data["comment_Key"])
            opinions_label.extend(batch_data["comment_info"]["opinion_label"].numpy().tolist())
            emotions_label.extend(batch_data["comment_info"]["emotion_label"].numpy().tolist())
            batch_opinion_classindex = torch.max(batch_data["comment_info"]["opinion_label"], 1)[1]
            opinions_label_classindex.extend(batch_opinion_classindex.numpy().tolist())
            batch_emotion_classindex = torch.max(batch_data["comment_info"]["emotion_label"], 1)[1]
            emotions_label_classindex.extend(batch_emotion_classindex.numpy().tolist())
            # predict
            output = net(
                batch_data
            )  # {'opinion_predict':tensor([[0.3335, 0.2738, 0.3926],[0.3429, 0.2711, 0.3860],[0.3410, 0.2730, 0.3860], [0.3411, 0.2715, 0.3874]], device='cuda:0', grad_fn=<SoftmaxBackward0>)'emotion_predict':tensor([[0.0940, 0.1774, 0.0932, 0.1248, 0.1377, 0.1427, 0.1352, 0.0952],}
            # process and save predict
            batch_opinion_predict = output.get("opinion_predict")
            opinion_preds.extend(batch_opinion_predict.cpu().numpy().tolist())
            batch_opinion_predict_classindex = torch.max(batch_opinion_predict, 1)[1]
            opinion_preds_classidx.extend(batch_opinion_predict_classindex.cpu().numpy().tolist())

            batch_emotion_predict = output.get("emotion_predict")
            emotion_preds.extend(batch_emotion_predict.cpu().numpy().tolist())
            batch_emotion_predict_classindex = torch.max(batch_emotion_predict, 1)[1]
            emotion_preds_classidx.extend(batch_emotion_predict_classindex.cpu().numpy().tolist())
            if args.mv_att == True:
                batch_mv_att = output.get("multiview_attention")
                multiview_attention.extend(batch_mv_att.cpu().numpy().tolist())

    opinion_precision_macro, opinion_recall_macro, opinion_f1_macro, _ = \
        metrics.precision_recall_fscore_support(opinions_label_classindex, opinion_preds_classidx, \
            labels=opinion_labels_list, average='macro')
    opinion_precision_micro, opinion_recall_micro, opinion_f1_micro, _  = \
        metrics.precision_recall_fscore_support(opinions_label_classindex, opinion_preds_classidx, \
            labels=opinion_labels_list, average='micro')
    opinion_precision_class, opinion_recall_class, opinion_f1_class, _  = \
        metrics.precision_recall_fscore_support(opinions_label_classindex, opinion_preds_classidx, \
            labels=opinion_labels_list, average=None)
    opinion_acc_score = metrics.accuracy_score(y_true=opinions_label_classindex, y_pred=opinion_preds_classidx)
    accuracy['opinion'] = {
        "macro": {
            "precision": opinion_precision_macro,
            "recal": opinion_recall_macro,
            "f1_score": opinion_f1_macro
        },
        "micro": {
            "precision": opinion_precision_micro,
            "recal": opinion_recall_micro,
            "f1_score": opinion_f1_micro
        },
        "class": {
            "precision": opinion_precision_class.tolist(),
            "recal": opinion_recall_class.tolist(),
            "f1_score": opinion_f1_class.tolist()
        },
        "accuracy": opinion_acc_score.tolist()
    }


    emotion_precision_macro, emotion_recall_macro, emotion_f1_macro, _ = \
        metrics.precision_recall_fscore_support(emotions_label_classindex, emotion_preds_classidx, \
            labels=emotion_labels_list, average='macro')
    emotion_precision_micro, emotion_recall_micro, emotion_f1_micro, _ = \
        metrics.precision_recall_fscore_support(emotions_label_classindex, emotion_preds_classidx, \
            labels=emotion_labels_list, average='micro')
    emotion_precision_class, emotion_recall_class, emotion_f1_class, _ = \
        metrics.precision_recall_fscore_support(emotions_label_classindex, emotion_preds_classidx, \
            labels=emotion_labels_list, average=None)
    emotion_acc_score = metrics.accuracy_score(y_true=emotions_label_classindex, y_pred=emotion_preds_classidx)
    accuracy['emotion'] = {
        "macro": {
            "precision": emotion_precision_macro,
            "recal": emotion_recall_macro,
            "f1_score": emotion_f1_macro
        },
        "micro": {
            "precision": emotion_precision_micro,
            "recal": emotion_recall_micro,
            "f1_score": emotion_f1_micro
        },
        "class": {
            "precision": emotion_precision_class.tolist(),
            "recal": emotion_recall_class.tolist(),
            "f1_score": emotion_f1_class.tolist()
        },
        "accuracy": emotion_acc_score.tolist()
    }
    # return 100 * np.mean(np.array(accuracy)), preds
    result = {}
    result["comments_key"] = comments_key
    result["opinion_preds"] = opinion_preds
    result["opinion_preds_classidx"] = opinion_preds_classidx
    result["emotion_preds"] = emotion_preds
    result["emotion_preds_classidx"] = emotion_preds_classidx
    if args.mv_att == True:
        result["multiview_attention"] = multiview_attention

    result["opinions_label"] = opinions_label
    result["opinions_label_classindex"] = opinions_label_classindex
    result["emotions_label"] = emotions_label
    result["emotions_label_classindex"] = emotions_label_classindex

    return accuracy, result