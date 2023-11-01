"""
This training procedure is an extended and adapted version of the one used in the SquiggleNet project, see
https://github.com/welch-lab/SquiggleNet/blob/master/trainer.py. For example, both steps (training and validation) got
an extra CrossEntropyLoss object which now balances the loss according to the number of reads per class.
"""

import click
import os
import glob
import pandas as pd
import time
import torch
import lightning.pytorch as pl

from dataloader import CustomDataLoader
from models.SquiggleNetModel import Bottleneck, SquiggleNetLightning
from models.DeepSelectNetModel import DSBottleneck, DeepSelectNet
from models.BNLSTM import bnLSTM
from MitoModel import VDCNN_bnLSTM_1window, VDCNN_gru_1window_hidden, regGru_32window_hidden_BN
from numpy import random
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch import nn
from tqdm import tqdm
from lightning.pytorch.callbacks import EarlyStopping, DeviceStatsMonitor, ModelSummary, ModelCheckpoint



PLASMID_LABEL = 0
CHR_LABEL = 1


def validate(out_folder, epoch, validation_generator, device, model, val_criterion):
    label_df = pd.DataFrame(columns=['Read ID', 'Predicted Label'])
    totals = dict.fromkeys(['Validation Loss', 'Validation Accuracy', 'TN', 'FP', 'FN', 'TP', 'Balanced Accuracy',
                            'F1S', 'Precision', 'Recall'], 0)

    # set gradient calculation off
    val_iterations = 0
    with torch.set_grad_enabled(False):
        for val_data, val_labels, val_ids in tqdm(validation_generator, desc='val-batch'):
            val_iterations += 1
            val_data, val_labels = val_data.to(device), val_labels.to(torch.long).to(device)
            val_outputs = model(val_data)
            val_loss = val_criterion(val_outputs, val_labels.unsqueeze(1).to(torch.float))
            totals['Validation Loss'] += val_loss.item()

            # get and store predicted labels
            #predicted_labels = val_outputs.max(dim=1).indices.int().data.cpu().numpy()
            predicted_labels = (val_outputs >= 0.5).type(torch.long).data.cpu().numpy()
            if None not in val_ids:
                for read_id, label_nr in zip(val_ids, predicted_labels):
                    label = 'plasmid' if label_nr == PLASMID_LABEL else 'chr'
                    label_df = pd.concat([label_df, pd.DataFrame([{'Read ID': read_id, 'Predicted Label': label}])],
                                         ignore_index=True)

            # calculate confusion matrix and performance metrics
            val_labels = val_labels.cpu().numpy()
            tn, fp, fn, tp = confusion_matrix(val_labels, predicted_labels, labels=[CHR_LABEL, PLASMID_LABEL]).ravel()
            totals['TN'] += tn
            totals['FP'] += fp
            totals['FN'] += fn
            totals['TP'] += tp
            totals['Validation Accuracy'] += accuracy_score(val_labels, predicted_labels)
            totals['Balanced Accuracy'] += balanced_accuracy_score(val_labels, predicted_labels)
            totals['F1S'] += f1_score(val_labels, predicted_labels, pos_label=PLASMID_LABEL)
            totals['Precision'] += precision_score(val_labels, predicted_labels, pos_label=PLASMID_LABEL)
            totals['Recall'] += recall_score(val_labels, predicted_labels, pos_label=PLASMID_LABEL)

    if not label_df.empty:
        label_df.to_csv(f'{out_folder}/pred_labels/pred_labels_epoch{epoch}.csv', index=False)

    return {k: v / val_iterations for k, v in totals.items()}


def update_stopping_criterion(current_loss, last_loss, trigger_times):
    if current_loss > last_loss:
        trigger_times += 1
    else:
        trigger_times = 0

    print(f'\nTrigger times: {str(trigger_times)}')
    return trigger_times


@click.command()
@click.option('--p_train', '-pt', help='folder with plasmid training tensor files', required=True,
              type=click.Path(exists=True))
@click.option('--p_val', '-pv', help='folder with plasmid validation tensor files', required=True,
              type=click.Path(exists=True))
@click.option('--p_ids', '-pid', help='file path of plasmid validation read ids', default=None, required=False)
@click.option('--chr_train', '-ct', help='folder with chromosome training tensor files', required=True)
              #type=click.Path(exists=True))
@click.option('--chr_val', '-cv', help='folder with chromosome validation tensor files', required=True,
              type=click.Path(exists=True))
@click.option('--chr_ids', '-cid', help='file path of chromosome validation read ids', default=None, required=False)
@click.option('--out_folder', '-o', help='output folder path in which logs and models are saved', required=True,
              type=click.Path())
@click.option('--interm', '-i', help='path to checkpoint file of already trained model (optional)', required=False,
              type=click.Path(exists=True))
@click.option('--patience', '-p', default=2, help='patience (i.e., number of epochs) to wait before early stopping')
@click.option('--batch_size', '-b', default=1000, help='batch size, default 1000 reads')
@click.option('--n_workers', '-w', default=4, help='number of workers, default 4')
@click.option('--n_epochs', '-e', default=5, help='number of epochs, default 5')
@click.option('--learning_rate', '-l', default=1e-3, help='learning rate, default 1e-3')
@click.option('--random_seed', '-s', default=42, help='random seed for file shuffling of custom data loaders')
@click.option('--train_model', '-m', default='SquiggleNet', help='deep learning model used for training (SquiggleNet, DeepSelectNet, bnLSTM, bnGRU, vdCNN_GRU, vdCNN_bnLSTM, regGRU)')
def main(p_train, p_val, p_ids, chr_train, chr_val, chr_ids, out_folder, interm, patience,
         batch_size, n_workers, n_epochs, learning_rate, random_seed, train_model):
    start_time = time.time()

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    if not os.path.exists(f'{out_folder}/logs'):
        os.makedirs(f'{out_folder}/logs')
    if not os.path.exists(f'{out_folder}/pred_labels'):
        os.makedirs(f'{out_folder}/pred_labels')
    if not os.path.exists(f'{out_folder}/models'):
        os.makedirs(f'{out_folder}/models')

    # load data
    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': n_workers,
              'pin_memory': False}  # speed up data transfer from CPU to GPU
    random_gen = random.default_rng(random_seed)

    transform_input = True
    if train_model == 'SquiggleNet' or train_model == 'DeepSelectNet':
        transform_input = False

    validation_generator = CustomDataLoader(p_val, chr_val, params, random_gen, p_ids, chr_ids, transform=transform_input)

    chr_dirs = glob.glob(f'{chr_train}_*')
    
    for chr_train_dir in chr_dirs:

        if chr_train_dir == f'{chr_train}_ALIGNED' or chr_train_dir == 'chr_train_13':
            continue
            
        if not os.path.exists(chr_train_dir):
            print("Error: " + chr_train_dir + " does not exist!")
            return

        training_generator = CustomDataLoader(p_train, chr_train_dir, params, random_gen, transform=transform_input)
    

        print(f'Class counts for training: {training_generator.get_class_counts()}')
        print(f'Class counts for validation: {validation_generator.get_class_counts()}')

        # create new or load trained model

        #if interm is not None:
        #    model.load_state_dict(torch.load(interm))

        # AdamW generalizes better and trains faster, see https://towardsdatascience.com/why-adamw-matters-736223f31b5d
        #optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        # use sample count per class for balancing the loss while training => only for Cross Entropy Loss
        # inspired by https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html
        #train_class_weights = [training_generator.get_n_reads() / (2 * class_count)
        #                       for class_count in training_generator.get_class_counts()]
        #train_class_weights = torch.as_tensor(train_class_weights, dtype=torch.float)
        #train_criterion = nn.CrossEntropyLoss(weight=train_class_weights).to(device)
    
        # Note JUU 18/10/2023
        # Calculate only weight for the positive class (plasmid) when using BCEwithLogitsLoss
        train_pos_weight = torch.as_tensor(training_generator.get_class_counts()[1] / (training_generator.get_class_counts()[0] + 1e-5), dtype=torch.float)

        # Note JUU 17/10/2023
        # Better use BCEWithLogitsLoss for binary classification
        # Note JUU 19/10/2023
        # implemented in Lightning Module Classes
        # train_criterion = nn.BCEWithLogitsLoss(pos_weight=train_pos_weight).to(device)
        

        # Note JUU 18/10/2023
        # Also for validation, better use BCEWithLogitsLoss for binary classification
        # with single weight for the positive class (plasmid)
        #val_class_weights = [validation_generator.get_n_reads() / (2 * class_count)
        #                     for class_count in validation_generator.get_class_counts()]
        #val_class_weights = torch.as_tensor(val_class_weights, dtype=torch.float)
        #val_criterion = nn.CrossEntropyLoss(weight=val_class_weights).to(device)


        val_pos_weight = torch.as_tensor(validation_generator.get_class_counts()[1] / (validation_generator.get_class_counts()[0] + 1e-5), dtype=torch.float)
        #val_criterion = nn.BCEWithLogitsLoss(pos_weight=val_pos_weight).to(device)
    
        if train_model == 'SquiggleNet':
            model = SquiggleNetLightning(Bottleneck, layers=[2, 2, 2, 2], learning_rate=learning_rate, batch_size=batch_size,train_pos_weight=train_pos_weight, val_pos_weight=val_pos_weight)
        elif train_model == 'DeepSelectNet':
            model = DeepSelectNet(DSBottleneck, layers=[2, 2, 2, 2], learning_rate=learning_rate, batch_size=batch_size,train_pos_weight=train_pos_weight, val_pos_weight=val_pos_weight)
        elif train_model == 'bnLSTM':
            model = bnLSTM(input_size=32, hidden_size=512, max_length=4000, learning_rate=learning_rate, batch_size=batch_size,train_pos_weight=train_pos_weight, val_pos_weight=val_pos_weight,
                        num_layers=1, use_bias=True, batch_first=True, dropout=0.5)
    #    elif train_model == 'vdCNN_bnLSTM':
    #        model = VDCNN_bnLSTM_1window(input_size=1, hidden_size=512, max_length=4000,  num_layers=1,
    #				 use_bias=True, batch_first=True, dropout=0.5)
    #    elif train_model == 'vdCNN_GRU':
    #        model = VDCNN_gru_1window_hidden(input_size=1, hidden_size=512, max_length=4000,  num_layers=1,
    #				 use_bias=True, batch_first=True, dropout=0.5)
    #    elif train_model == 'regGRU':
    #        model = regGru_32window_hidden_BN(input_size=1, hidden_size=512, max_length=4000,  num_layers=1,
    #				 use_bias=True, batch_first=True, dropout=0.5)
        else:
            print("Wrong model definition: " + str(train_model))
            return


        chkpath = str(out_folder) + "/models/"
        all_models = glob.glob(f'{chkpath}/epoch*-val_loss*-val_bal_acc*.ckpt')
        best_model = ""
        best_acc = 0.0
        for f in all_models:
            bacc = float((f.split("val_bal_acc")[1]).split(".ckpt")[0])
            if bacc > best_acc:
                best_acc = bacc
                best_model = f
        
        early_stop_callback = EarlyStopping(monitor="val_bal_acc", min_delta=0.00, patience=10, verbose=False, mode="max")
        device_stats = DeviceStatsMonitor()
        model_summary = ModelSummary(max_depth=-1)
        
        checkpoint = ModelCheckpoint(monitor='val_bal_acc', dirpath=chkpath, filename='epoch{epoch:02d}-val_loss{val_loss:.2f}-val_bal_acc{val_bal_acc:.2f}', mode="max", auto_insert_metric_name=False, save_top_k=1) 
        # callbacks=[early_stop_callback, device_stats],
        trainer = pl.Trainer(default_root_dir=out_folder, max_epochs=n_epochs, callbacks=[checkpoint, early_stop_callback])
        if len(best_model) > 0:
            trainer.fit(model, training_generator, validation_generator, ckpt_path=best_model)
        else:
            trainer.fit(model, training_generator, validation_generator)

    return

    # setup best model consisting of epoch and metric (for accuracy and loss as model selection criterion)
    best_model_acc = (0, 0)
    best_model_loss = (0, 1)

    # setup early stopping
    last_loss = 1.0
    trigger_times = 0

    train_results = pd.DataFrame(columns=['Epoch', 'Batch', 'Training Loss', 'Training Accuracy'])
    val_results = pd.DataFrame(columns=['Epoch', 'Validation Loss', 'Validation Accuracy', 'TN', 'FP', 'FN', 'TP',
                                        'Balanced Accuracy', 'F1S', 'MCC', 'Precision', 'Recall'])

    for epoch in tqdm(range(n_epochs), desc='epoch'):

        # Note JUU 13/10/2023
        # This was not set in the original SquiggleNet implementation and Nina's code
        # if not set correctly will impact BatchNorm and DropOut layers
        model.train(True)

        for i, (train_data, train_labels, _) in tqdm(enumerate(training_generator), desc='train-batch'):
            train_data, train_labels = train_data.to(device), train_labels.to(torch.float).to(device)

            # perform forward propagation
            outputs_train = model(train_data)
            train_loss = train_criterion(outputs_train, train_labels.unsqueeze(1))
            pred_labels = (outputs_train >= 0.5).type(torch.float)
            #print(pred_labels)
            train_acc = (train_labels == pred_labels).float().mean().item()
            #print(train_acc)
            train_results = pd.concat(
                [train_results, pd.DataFrame([{'Epoch': epoch, 'Batch': i, 'Training Loss': train_loss.item(),
                                               'Training Accuracy': train_acc}])], ignore_index=True)

            # perform backward propagation
            # -> set gradients to zero (to avoid using combination of old and new gradient as new gradient)
            optimizer.zero_grad()
            # -> compute gradients of loss w.r.t. model parameters
            train_loss.backward()
            # -> update parameters of optimizer
            optimizer.step()

        # Note JUU 13/10/2023
        # This was not set in the original SquiggleNet implementation and Nina's code
        # if not set correctly will impact BatchNorm and DropOut layers
        model.eval()
        # validate
        current_val_results = validate(out_folder, epoch, validation_generator, device, model, val_criterion)
        print(f'\nValidation Loss: {str(current_val_results["Validation Loss"])}, '
              f'Validation Accuracy: {str(current_val_results["Validation Accuracy"])}')
        current_val_results['Epoch'] = epoch
        val_results = pd.concat([val_results, pd.DataFrame([current_val_results])], ignore_index=True)

        # save logs and model per epoch
        train_results.to_csv(f'{out_folder}/logs/train_results_epoch{epoch}.csv', index=False)
        val_results.to_csv(f'{out_folder}/logs/val_results_epoch{epoch}.csv', index=False)
        torch.save(model.state_dict(), f'{out_folder}/models/model_epoch{epoch}.pt')

        # update best models
        if best_model_acc[1] < current_val_results['Validation Accuracy']:
            best_model_acc = (epoch, current_val_results['Validation Accuracy'])
        if best_model_loss[1] > current_val_results['Validation Loss']:
            best_model_loss = (epoch, current_val_results['Validation Loss'])

        # avoid overfitting with early stopping
        trigger_times = update_stopping_criterion(current_val_results['Validation Loss'], last_loss, trigger_times)
        last_loss = current_val_results['Validation Loss']
        if trigger_times >= patience:
            print(f'\nTraining would be early stopped!')
            # return  # TODO: comment in again if early stopping criterion is optimized

        # log current best model and runtime
        if epoch != (n_epochs - 1):
            print(f'\nCurrent best model based on accuracy: epoch {best_model_acc[0]}, value {best_model_acc[1]}\n'
                  f'Current best model based on loss: epoch {best_model_loss[0]}, value {best_model_loss[1]}\n'
                  f'Current runtime: {time.time() - start_time} seconds')

    print(f'\nOverall best model based on accuracy: epoch {best_model_acc[0]}, value {best_model_acc[1]}\n'
          f'Overall best model based on loss: epoch {best_model_loss[0]}, value {best_model_loss[1]}\n'
          f'Overall runtime: {time.time() - start_time} seconds')


if __name__ == '__main__':
    main()
