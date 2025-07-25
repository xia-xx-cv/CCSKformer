import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from train import seed_everything
import numpy as np
import pandas as pd
import time
from dataset import TwoviewImgDataset
from sklearn.metrics import roc_auc_score
import torchvision.transforms as transform
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import models.create_models as create
from tqdm import tqdm

seed_everything(111279)

def test_model2(netname, model, test_df, device):
    n_class = 5

    transform_test = transform.Compose([
        transform.ToPILImage(),
        transform.Resize((224, 224)),
        transform.ToTensor(),
        # transform.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    test_dataset = TwoviewImgDataset(TEST_PATH, test_df, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=1, pin_memory=False)
    dataset_size = len(test_loader)
    print('dataset_size', dataset_size)
    since = time.time()

    # Initialize lists to store all the true labels and predicted probabilities
    all_labels = []
    all_outputs = []

    # Initialize metrics
    roc_matrix = pd.DataFrame(columns=['label', '0', '1', '2', '3', '4'])
    total_outs, labels = torch.tensor([]), torch.tensor([])
    FM = np.zeros((n_class, n_class))  # confusion matrix
    tp = [0] * n_class
    tn = [0] * n_class
    fp = [0] * n_class
    fn = [0] * n_class
    precision = [0] * n_class
    recall = [0] * n_class  # i.e., sensitivity
    specificity = [0] * n_class
    f1 = [0] * n_class
    running_loss = 0.0
    running_corrects = 0

    criterion = nn.CrossEntropyLoss(reduction='sum')
    valid_bar = tqdm(test_loader)
    # -------------- predicting ---------------
    for i, (img, label) in enumerate(valid_bar):
        img = img.to(device, non_blocking=True)
        label = label.to(device)
        B, V, C, H, W = img.size()
        img = img.view(-1, C, H, W)

        with torch.no_grad():
            output, _ = model(img)

        outputs = output[0] + output[1]
        loss = criterion(outputs, label)

        labels = torch.concat((labels, label), dim=0) if labels.size(0) > 0 else label
        total_outs = torch.concat((total_outs, outputs.detach()), dim=0) \
            if total_outs.size(0) > 0 else outputs.detach()

        running_loss += loss.item()


        all_labels.extend(label.cpu().numpy())
        all_outputs.extend(outputs.softmax(dim=1).cpu().numpy())
    outputs_softmax = torch.softmax(total_outs, 1).cpu()
    preds = torch.argmax(outputs_softmax, dim=1)
    cls_statis = torch.bincount(labels, minlength=5)
    # cls_statis = [torch.sum(labels == 0), torch.sum(labels == 1), sum(labels == 2), sum(labels == 3), sum(labels == 4)]
    running_corrects += torch.sum(preds == labels)
    # -------------- computing metrics ---------------
    for batch_i in range(len(labels)):
        roc_df = {'label': labels[batch_i],
                  '0': outputs_softmax[batch_i][0],
                  '1': outputs_softmax[batch_i][1],
                  '2': outputs_softmax[batch_i][2],
                  '3': outputs_softmax[batch_i][3],
                  '4': outputs_softmax[batch_i][4]
                  }
        roc_matrix = roc_matrix._append(roc_df, ignore_index=True)

        predict_label = preds[batch_i]
        true_label = labels[batch_i]
        FM[true_label][predict_label] = FM[true_label][predict_label] + 1

        for label in range(n_class):
            p_or_n_from_pred = (label == preds[batch_i])
            p_or_n_from_label = (label == labels[batch_i])

            if p_or_n_from_pred == 1 and p_or_n_from_label == 1:
                tp[label] += 1
            if p_or_n_from_pred == 0 and p_or_n_from_label == 0:
                tn[label] += 1
            if p_or_n_from_pred == 1 and p_or_n_from_label == 0:
                fp[label] += 1
            if p_or_n_from_pred == 0 and p_or_n_from_label == 1:
                fn[label] += 1

    for label in range(n_class):
        precision[label] = tp[label] / (tp[label] + fp[label] + 1e-8)
        recall[label] = tp[label] / (tp[label] + fn[label] + 1e-8)
        specificity[label] = tn[label] / (tn[label] + fp[label] + 1e-8)
        f1[label] = 2 * precision[label] * recall[label] / (precision[label] + recall[label] + 1e-8)

        print('Class {}: \t Precision: {:.4f}, Recall: {:.4f}, Specificity: {:.4f}, F1:{:.4f}'.format(
            label, precision[label], recall[label], specificity[label], f1[label]))
        fileHandle = open(netname + '_test_result.txt', 'a')
        fileHandle.write('Class {}: \t Precision: {:.4f}, Recall: {:.4f}, Specificity: {:.4f}, F1:{:.4f}\n'.format(
            label, precision[label], recall[label], specificity[label], f1[label]))
        fileHandle.close()

    print('\nConfusion Matrix:')
    print(FM)
    fileHandle = open(netname + '_test_result.txt', 'a')
    fileHandle.write('\nConfusion Matrix:\n')
    for f_i in FM:
        fileHandle.write(str(f_i) + '\r\n')
    fileHandle.close()

    roc_matrix.to_csv(netname + '_roc_data.csv', encoding='gbk')

    pe = np.sum((np.array(tp)+np.array(fn))*(np.array(tp)+np.array(fp)))/(dataset_size * dataset_size)
    pa = np.sum(tp) / dataset_size
    # pe0 = (tp[0] + fn[0]) * (tp[0] + fp[0])
    # pe1 = (tp[1] + fn[1]) * (tp[1] + fp[1])
    # pe2 = (tp[2] + fn[2]) * (tp[2] + fp[2])
    # pe3 = (tp[3] + fn[3]) * (tp[3] + fp[3])
    # pe4 = (tp[4] + fn[4]) * (tp[4] + fp[4])
    # pe = (pe0 + pe1 + pe2 + pe3 + pe4) / (dataset_size * dataset_size)
    # pa = (tp[0] + tp[1] + tp[2] + tp[3] + tp[4]) / dataset_size
    kappa = (pa - pe) / (1 - pe)

    test_epoch_loss = running_loss / dataset_size
    test_epoch_acc = running_corrects / dataset_size
    overall_precision = sum([cls_statis[i] * p for i, p in enumerate(precision)]) / sum(cls_statis)
    overall_recall = sum([cls_statis[i] * r for i, r in enumerate(recall)]) / sum(cls_statis)
    overall_specificity = sum([cls_statis[i] * s for i, s in enumerate(specificity)]) / sum(cls_statis)
    overall_f1 = sum([cls_statis[i] * f for i, f in enumerate(f1)]) / sum(cls_statis)

    # Calculate and print AUC
    all_labels = np.array(all_labels)
    all_outputs = np.array(all_outputs)
    aucs = {}
    auc_sum = 0
    for i in range(n_class):
        auc = roc_auc_score(all_labels == i, all_outputs[:, i])
        aucs[i] = auc
        auc_sum += auc
        print(f'Class {i} AUC: {auc:.4f}')
        fileHandle = open(netname + '_test_result.txt', 'a')
        fileHandle.write(f'Class {i} AUC: {auc:.4f}\n')
        fileHandle.close()

    # Calculate average AUC
    avg_auc = auc_sum / n_class
    print(f'Average AUC: {avg_auc:.4f}')
    fileHandle = open(netname + '_test_result.txt', 'a')
    fileHandle.write(f'Average AUC: {avg_auc:.4f}\n')
    fileHandle.close()

    elapsed_time = time.time() - since
    print(
        'Test loss: {:.4f}, acc: {:.4f}, kappa: {:.4f}, avg_precision: {:.4f}, avg_recall: {:.4f}, avg_specificity: {:.4f}, avg_f1: {:.4f}, Total elapsed time: {:.4f} '.format(
            test_epoch_loss, test_epoch_acc, kappa, overall_precision, overall_recall, overall_specificity, overall_f1,
            elapsed_time))
    fileHandle = open(netname + '_test_result.txt', 'a')
    fileHandle.write('Test loss: {:.4f}, acc: {:.4f}, kappa: {:.4f}, avg_precision: {:.4f}, avg_recall: {:.4f}, avg_specificity: {:.4f}, avg_f1: {:.4f}, Total elapsed time: {:.4f}\n'.format(
        test_epoch_loss, test_epoch_acc, kappa, overall_precision, overall_recall, overall_specificity, overall_f1, elapsed_time))
    fileHandle.close()


if __name__ == '__main__':
  # please run the preprocess/DataEnhance/py first before running this file
    split = "test"
    DATA_PATH = "yourpath/aptos2019/"
    TEST_PATH = os.path.join(DATA_PATH, '{}_preprocessed'.format(split))
    BATCH_SIZE = 1
    MODELPATH = "weight/ccsk_1e-05_100_d12_h9_MKT_AdaptSWT-0.4331.pth"
    checkpoint = torch.load(MODELPATH, weights_only=True, map_location='cpu')
    if isinstance(checkpoint, dict):
        model = create.my_CCSKFormer(
            view=2,
            pretrained=True,
            num_classes=5,
            pre_Path=MODELPATH,
            depth=12,
            num_heads=9,
            # embed_dim=768,
            drop_rate=0.1,
            drop_path_rate=0.1,
        )
    else:
        model = checkpoint
    test_csv_path = os.path.join(DATA_PATH, 'test2view.csv'.format(split))
    test_df = pd.read_csv(test_csv_path)
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()

    test_model2(netname='ccsk', model=model, test_df=test_df, device=device)
