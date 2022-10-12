import click
import time
import torch
import os

from dataset import Dataset
from model import Bottleneck, ResNet
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, \
    precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader


def validate(validation_generator, device, model, criterion):
    totals = dict.fromkeys(['Validation Loss', 'TN', 'FP', 'FN', 'TP', 'Validation Accuracy', 'Balanced Accuracy',
                            'F1S', 'MCC', 'Precision', 'Recall'], 0)

    # set gradient calculation off
    with torch.set_grad_enabled(False):
        for val_data, val_labels in validation_generator:
            val_data, val_labels = val_data.to(device), val_labels.to(torch.long).to(device)
            val_outputs = model(val_data)

            val_loss = criterion(val_outputs, val_labels)
            totals['Validation Loss'] += val_loss.item()

            # calculate confusion matrix and performance metrics
            # TODO: once final metric is chosen, reduce number of calculated metrics
            predicted_labels = val_outputs.max(dim=1).indices.int().data.cpu().numpy()
            tn, fp, fn, tp = confusion_matrix(val_labels, predicted_labels, labels=[1, 0]).ravel()
            totals['TN'] += tn
            totals['FP'] += fp
            totals['FN'] += fn
            totals['TP'] += tp

            totals['Validation Accuracy'] += accuracy_score(val_labels, predicted_labels)
            totals['Balanced Accuracy'] += balanced_accuracy_score(val_labels, predicted_labels)
            totals['F1S'] += f1_score(val_labels, predicted_labels, pos_label=0)
            totals['MCC'] += matthews_corrcoef(val_labels, predicted_labels)
            totals['Precision'] += precision_score(val_labels, predicted_labels, pos_label=0)
            totals['Recall'] += recall_score(val_labels, predicted_labels, pos_label=0)

    return {k: v / len(validation_generator) for k, v in totals.items()}


def update_stopping_criterion(current_loss, last_loss, trigger_times):
    if current_loss > last_loss:
        trigger_times += 1
    else:
        trigger_times = 0

    print(f'Trigger times: {str(trigger_times)}')
    return trigger_times


@click.command()
@click.option('--p_train', '-pt', help='file path of plasmid training set', type=click.Path(exists=True))
@click.option('--p_val', '-pv', help='file path of plasmid validation set', type=click.Path(exists=True))
@click.option('--chr_train', '-ct', help='file path of chromosome training set', type=click.Path(exists=True))
@click.option('--chr_val', '-cv', help='file path of chromosome validation set', type=click.Path(exists=True))
@click.option('--out_folder', '-o', help='output folder path in which models are saved', type=click.Path())
@click.option('--interm', '-i', help='file path of model checkpoint (optional)', type=click.Path(exists=True),
              required=False)
@click.option('--model_selection_criterion', '-s', default='loss', type=click.Choice(['loss', 'acc']),
              help='model selection criterion, choose between validation loss ("loss") and validation accuracy ("acc")')
@click.option('--patience', '-p', default=2, help='patience (i.e., number of epochs) to wait before early stopping')
@click.option('--batch', '-b', default=1000, help='batch size, default 1000 reads')
@click.option('--n_workers', '-w', default=8, help='number of workers, default 8')
@click.option('--n_epochs', '-e', default=5, help='number of epochs, default 5')
@click.option('--learning_rate', '-l', default=1e-3, help='learning rate, default 1e-3')
def main(p_train, p_val, chr_train, chr_val, out_folder, interm, model_selection_criterion, patience, batch, n_workers,
         n_epochs, learning_rate):
    start_time = time.time()

    if model_selection_criterion not in ['loss', 'acc']:
        raise ValueError('Model selection criterion (-s) must be "loss" or "acc"!')

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # load data
    params = {'batch_size': batch,
              'shuffle': True,
              'num_workers': n_workers}
    training_set = Dataset(p_train, chr_train)
    training_generator = DataLoader(training_set, **params)
    validation_set = Dataset(p_val, chr_val)
    validation_generator = DataLoader(validation_set, **params)

    print(f'Number of batches: {str(len(training_generator))}')

    # create new or load pre-trained model
    model = ResNet(Bottleneck, layers=[2, 2, 2, 2]).to(device)
    if interm is not None:
        model.load_state_dict(torch.load(interm))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # use sample count per class for balancing the loss while training
    # inspired by https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html
    class_weights = [len(training_set) / (2 * class_count) for class_count in training_set.get_class_counts()]
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)

    # setup best model consisting of epoch and metric (acc/ loss)
    if model_selection_criterion == 'acc':
        best_model = (0, 0)
    else:
        best_model = (0, 1)

    # setup early stopping
    last_loss = 1.0
    trigger_times = 0

    for epoch in range(n_epochs):
        print(f'\nEpoch: {str(epoch)}')

        for i, (train_data, train_labels) in enumerate(training_generator):
            train_data, train_labels = train_data.to(device), train_labels.to(torch.long).to(device)

            # perform forward propagation
            outputs_train = model(train_data)
            train_loss = criterion(outputs_train, train_labels)
            train_acc = 100.0 * (train_labels == outputs_train.max(dim=1).indices).float().mean().item()
            print(f'Batch: {str(i)}, Training Loss: {str(train_loss.item())}, Training Accuracy: {str(train_acc)}')

            # perform backward propagation
            # -> set gradients to zero (to avoid using combination of old and new gradient as new gradient)
            optimizer.zero_grad()
            # -> compute gradients of loss w.r.t. model parameters
            train_loss.backward()
            # -> update parameters of optimizer
            optimizer.step()

        # validate and log results
        val_results = validate(validation_generator, device, model, criterion)
        print(f'Validation: {val_results}')

        # save each model
        torch.save(model.state_dict(), f'{out_folder}/model_epoch{epoch}.pt')

        # update best model
        if model_selection_criterion == 'acc':
            if best_model[1] < val_results['Validation Accuracy']:
                best_model = (epoch, val_results['Validation Accuracy'])
        else:
            if best_model[1] > val_results['Validation Loss']:
                best_model = (epoch, val_results['Validation Loss'])

        # avoid overfitting with early stopping
        trigger_times = update_stopping_criterion(val_results['Validation Loss'], last_loss, trigger_times)
        last_loss = val_results['Validation Loss']

        if trigger_times >= patience:
            print(f'Training would be early stopped!\n'
                  f'Best model would be reached after {str(best_model[0])} epochs '
                  f'with runtime of {time.time() - start_time} seconds')
            # return  # TODO: comment in again if early stopping criterion is optimized

    print(f'Best model reached after epoch no. {str(best_model[0])}\n '
          f'Runtime: {time.time() - start_time} seconds')


if __name__ == '__main__':
    main()
