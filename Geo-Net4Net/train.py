import torch
import os
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from tqdm import tqdm
from torchsummary import summary
import numpy as np
from GeneratePseudoLabels import GeneratePseudoLabels

import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

from dataset import FCDataset, ReassignedFCDataset
from SPD_DNN.optimizer import StiefelMetaOptimizer
from net import MSNet
from MS_RNN import cosine_similarity, similarity_loss
from utils import cluster_score

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if __name__ == '__main__':

    num_classes = 12
    windows = [40]
    brain_size = 268
    lr = 0.01
    scans = ['scan1', 'scan2']

    resume = None
    resume_log = None

    batch_size = 1
    num_workers = 8


    use_cuda = False
    save_result = True
    total_epochs = 100

    input_dir = 'input'
    label_dir = 'labels.csv'
    output_dir = 'output'

    for window in windows:
        print('\n############### current window = %d ###############\n' % window)
        result_path = os.path.join('train_results', output_dir,
                                   '{}/window={}'.format('mix' if isinstance(scans, list) else scans, window))

        device = torch.device('cuda:0' if use_cuda else 'cpu')

        train_dataset = test_dataset = val_dataset = None
        for scan in scans:
            scan_dir = os.path.join(input_dir, scan)

            input_window_dir = os.path.join(scan_dir, 'window=%d' % window)
            train_data_dir = os.path.join(input_window_dir, 'train')
            test_data_dir = os.path.join(input_window_dir, 'test')
            val_data_dir = os.path.join(input_window_dir, 'val')
            models_save_path = os.path.join(result_path, 'models_save')

            train_dataset = FCDataset(data_dir=train_data_dir, label_dir=label_dir) \
                if train_dataset is None else train_dataset + FCDataset(data_dir=train_data_dir, label_dir=label_dir)

            test_dataset = FCDataset(data_dir=test_data_dir, label_dir=label_dir) \
                if test_dataset is None else test_dataset + FCDataset(data_dir=test_data_dir, label_dir=label_dir)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

            val_dataset = FCDataset(data_dir=val_data_dir, label_dir=label_dir) \
                if val_dataset is None else val_dataset + FCDataset(data_dir=val_data_dir, label_dir=label_dir)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        model = MSNet(num_classes=num_classes)

        print(result_path)

        if resume:
            print('resume: ' + resume)
            model.load_state_dict(torch.load(resume))

        if not os.path.exists(models_save_path):
            os.makedirs(models_save_path)

        if save_result:
            with open(os.path.join(result_path, 'net.txt'), 'w') as f:
                a = torch.randn(8, brain_size, brain_size)
                a = a @ a.transpose(-2, -1) + 1e-4 * torch.eye(brain_size)
                model_str = str(summary(model, a, device='cuda:0' if use_cuda else 'cpu', depth=4))
                f.write(model_str)

        if use_cuda:
            model = model.cuda()
        else:
            model = model.cpu()

        criterion = similarity_loss
        optimizer = SGD(model.parameters(), lr=lr)
        optimizer = StiefelMetaOptimizer(optimizer)

        @torch.no_grad()
        def generate_pseudo_labels(dataset):
            model.eval()
            bar = tqdm(enumerate(dataset), total=len(dataset))
            pseudo_labels = []
            for batch_idx, (inputs, targets) in bar:
                inputs = inputs.squeeze()

                if use_cuda:
                    inputs = inputs.cuda()

                outputs = model(inputs)

                labels_pred = GeneratePseudoLabels(n_clusters=num_classes, bandwidth=50).fit(outputs).labels_
                pseudo_labels.append(torch.from_numpy(labels_pred))

                bar.set_description('arrange pseudo labels')
            return ReassignedFCDataset(dataset, pseudo_labels)

        # Training
        def train(data_loader):
            model.train()
            train_loss = 0
            total_purity = 0.0
            total_nmi = 0.0
            total = 0.0
            bar = tqdm(enumerate(data_loader), total=len(data_loader))
            for batch_idx, (inputs, targets, pseudo_labels) in bar:
                if inputs.isnan().any():
                    print('error', batch_idx)
                    continue
                inputs = inputs.squeeze()
                targets = targets.squeeze()

                if use_cuda:
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                    pseudo_labels = pseudo_labels.cuda()

                optimizer.zero_grad()
                outputs = model(inputs)

                loss = criterion(outputs, pseudo_labels, 'spd_dist')

                loss.backward()
                optimizer.step()

                labels_pred = GeneratePseudoLabels(n_clusters=num_classes, bandwidth=50).fit(outputs).labels_

                purity, _, nmi = cluster_score(targets.cpu(), labels_pred)

                train_loss += loss.data.item()
                total += targets.size(0)
                total_purity += purity
                total_nmi += nmi

                bar.set_description('Loss: %.4f | purity: %.4f | nmi: %.4f'
                                    % (train_loss / (batch_idx + 1.0), total_purity / (batch_idx + 1.0),
                                       total_nmi / (batch_idx + 1.0)))
            return train_loss / len(data_loader), total_purity / len(data_loader)

        @torch.no_grad()
        def test(data_loader):
            model.eval()
            bar = tqdm(enumerate(data_loader), total=len(data_loader))
            test_loss = 0
            total_purity = 0.0
            total_nmi = 0.0
            total = 0.0
            for batch_idx, (inputs, targets) in bar:
                inputs = inputs.squeeze()
                targets = targets.squeeze()

                if use_cuda:
                    inputs = inputs.cuda()
                    targets = targets.cuda()

                outputs = model(inputs)
                loss = criterion(outputs, targets, 'spd_dist')

                labels_pred = GeneratePseudoLabels(n_clusters=num_classes, bandwidth=50).fit(outputs).labels_

                purity, _, nmi = cluster_score(targets, labels_pred)

                test_loss += loss.data.item()
                total += targets.size(0)
                total_purity += purity
                total_nmi += nmi

                bar.set_description('Loss: %.4f | purity: %.4f | nmi: %.4f'
                                    % (test_loss / (batch_idx + 1.0), total_purity / (batch_idx + 1.0),
                                       total_nmi / (batch_idx + 1.0)))

            return (test_loss / len(data_loader), total_purity / len(data_loader))

        train_losses = []
        train_accs = []

        test_losses = []
        test_accs = []

        val_losses = []
        val_accs = []

        epochs = []
        best_acc = 0

        start_epoch = 1

        if resume_log:
            print('resume_log: ' + resume_log)
            log = np.loadtxt(resume_log, delimiter=',')
            train_losses, train_accs = log[:, 1].tolist(), log[:, 2].tolist()
            test_losses, test_accs = log[:, 3].tolist(), log[:, 4].tolist()
            val_losses, val_accs = log[:, 5].tolist(), log[:, 6].tolist()
            epochs = log[:, 0].astype(np.int).tolist()
            best_acc = max(test_accs)
            start_epoch = max(epochs) + 1

            for i, epoch in enumerate(epochs):
                print('\nEpoch: %d' % epoch)
                print('Loss: %.4f | purity: %.4f' % (train_losses[i], train_accs[i]))
                print('Loss: %.4f | purity: %.4f' % (test_losses[i], test_accs[i]))
                print('Loss: %.4f | purity: %.4f' % (val_losses[i], val_accs[i]))

        for epoch in range(start_epoch, start_epoch + total_epochs):
            print('\nEpoch: %d' % epoch)
            new_train_dataset = generate_pseudo_labels(train_dataset)
            train_loader = DataLoader(new_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            train_loss, train_acc = train(train_loader)
            test_loss, test_acc = test(test_loader)
            val_loss, val_acc = test(val_loader)

            if test_acc > best_acc:
                print('best')
                best_acc = test_acc

            epochs.append(epoch)

            train_losses.append(train_loss)
            train_accs.append(train_acc)
            min_train_loss = np.min(train_losses)
            min_train_loss_idx = np.argmin(train_losses) + 1
            max_train_acc = np.max(train_accs)
            max_train_acc_idx = np.argmax(train_accs) + 1

            test_losses.append(test_loss)
            test_accs.append(test_acc)
            min_test_loss = np.min(test_losses)
            min_test_loss_idx = np.argmin(test_losses) + 1
            max_test_acc = np.max(test_accs)
            max_test_acc_idx = np.argmax(test_accs) + 1

            val_losses.append(val_loss)
            val_accs.append(val_acc)
            min_val_loss = np.min(val_losses)
            min_val_loss_idx = np.argmin(val_losses) + 1
            max_val_acc = np.max(val_accs)
            max_val_acc_idx = np.argmax(val_accs) + 1

            if save_result:
                plt.figure(1)

                plt.plot(epochs, train_losses, label='train loss')
                plt.plot(min_train_loss_idx, min_train_loss, 'ko')
                plt.annotate('(%d,%.3f)' % (min_train_loss_idx, min_train_loss),
                             xy=(min_train_loss_idx, min_train_loss),
                             xytext=(min_train_loss_idx, min_train_loss))

                plt.plot(epochs, test_losses, label='test loss')
                plt.plot(min_test_loss_idx, min_test_loss, 'ko')
                plt.annotate('(%d,%.3f)' % (min_test_loss_idx, min_test_loss), xy=(min_test_loss_idx, min_test_loss),
                             xytext=(min_test_loss_idx, min_test_loss))

                plt.plot(epochs, val_losses, label='val loss')
                plt.plot(min_val_loss_idx, min_val_loss, 'ko')
                plt.annotate('(%d,%.3f)' % (min_val_loss_idx, min_val_loss), xy=(min_val_loss_idx, min_val_loss),
                             xytext=(min_val_loss_idx, min_val_loss))

                plt.legend()
                plt.xlabel('epoch')
                plt.ylabel('loss')
                plt.savefig(os.path.join(result_path, 'loss.svg'))
                plt.close()

                plt.figure(2)

                plt.plot(epochs, train_accs, label='train purity')
                plt.plot(max_train_acc_idx, max_train_acc, 'ko')
                plt.annotate('(%d,%.3f)' % (max_train_acc_idx, max_train_acc), xy=(max_train_acc_idx, max_train_acc),
                             xytext=(max_train_acc_idx, max_train_acc))

                plt.plot(epochs, test_accs, label='test purity')
                plt.plot(max_test_acc_idx, max_test_acc, 'ko')
                plt.annotate('(%d,%.3f)' % (max_test_acc_idx, max_test_acc), xy=(max_test_acc_idx, max_test_acc),
                             xytext=(max_test_acc_idx, max_test_acc))

                plt.plot(epochs, val_accs, label='val purity')
                plt.plot(max_val_acc_idx, max_val_acc, 'ko')
                plt.annotate('(%d,%.3f)' % (max_val_acc_idx, max_val_acc), xy=(max_val_acc_idx, max_val_acc),
                             xytext=(max_val_acc_idx, max_val_acc))

                plt.legend()
                plt.xlabel('epoch')
                plt.ylabel('purity')
                plt.savefig(os.path.join(result_path, 'purity.svg'))
                plt.close()

                with open(os.path.join(result_path, 'log.txt'), 'a') as log_file:
                    log_file.write(
                        '%d,%f,%f,%f,%f,%f,%f\n' % (
                            epoch, train_loss, train_acc, test_loss, test_acc, val_loss, val_acc))
                torch.save(model.state_dict(), os.path.join(models_save_path, "%d_%.3f.pth" % (epoch, test_acc)))
