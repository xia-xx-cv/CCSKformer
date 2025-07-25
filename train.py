import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random
import numpy as np
import pandas as pd
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transform
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CosineAnnealingLR
from torch.utils.data import DataLoader
from dataset import SingleImgDataset, TwoviewImgDataset
import models.create_models as create
from models.FLoss import FocalLoss


def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results
    Arguments:
        seed {int}
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(model_name ,N_EPOCHS=100 ,LR = 0.0001 ,depth=12 ,head=9):
    print(model_name)
    model = create.my_CCSKFormer(
        pretrained=True,
        num_classes=5,
        pre_Path = 'weights/ccsk_1e-05_100_d12_h9_MKT_AdaptSWT-0.4331.pth',
        depth=depth,
        num_heads=head,
        view=2,
        # embed_dim=768,
        drop_rate=0.1,
        drop_path_rate=0.1,
    )

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=4, T_mult=2)
    # scheduler = CosineAnnealingLR(optimizer,T_max=50)
    # criterion = nn.CrossEntropyLoss()
    criterion = FocalLoss(gamma=2,alpha=0.25)

    train_loss = []
    train_acc = []
    train_acc_2 =[]
    train_loss_2 = []
    valid_loss = []
    valid_acc = []
    inter_val = 1

    best_model = copy.deepcopy(model)
    # last_model = model
    model.to(device)

    best_acc = 0
    best_test = 0
    for epoch in range(N_EPOCHS):

        train_epoch_loss = 0.0
        train_epoch_acc = 0.0

        model.train()
        train_bar = tqdm(train_loader)

        for i, (img, label) in enumerate(train_bar):
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            B, V, C, H, W = img.size()
            imgs_mix = img.view(-1, C, H, W)
            # b,v = label.shape
            # label=label.view(b*v)
            optimizer.zero_grad()
            output, _ = model(imgs_mix)
            output = output[1] + output[0]
            loss = criterion(output, label)
            loss.backward()
            train_epoch_acc += (output.detach().argmax(dim=1) == label).sum()
            train_epoch_loss += loss.item()

            optimizer.step()
            scheduler.step()

        train_loss_mean = train_epoch_loss / len(train_loader)
        train_acc_mean = train_epoch_acc / (len(train_dataset) * NUM_VIEW)

        train_loss.append(train_loss_mean)
        train_acc.append(train_acc_mean)

        print('{} train loss: {:.3f} train acc: {:.3f}  lr:{}'.format(epoch, train_loss_mean,
                                                                  train_acc_mean*2,
                                                                      optimizer.param_groups[-1]['lr']))
        if (epoch + 1) % inter_val == 0:
            val_acc_mean, val_loss_mean = val_model(model, valid_loader, criterion, device)

            if val_acc_mean > best_acc:
                model.cpu()
                best_model = copy.deepcopy(model.state_dict())
                model.to(device)
                best_acc = val_acc_mean

            valid_loss.append(val_loss_mean)
            valid_acc.append(val_acc_mean.cpu())

        # test_acc_mean = testModel(model,test_loader,len(test_dataset),device)
        # if test_acc_mean>best_test:
        #     best_test = test_acc_mean
        torch.cuda.empty_cache()
    print("best val acc:", best_acc)
    if best_test != 0: print("best test acc:", best_test)
    torch.save(best_model, os.path.join(SAVE_PT_DIR, '{}-{:.4f}.pth'.format(model_name, best_acc)))

    # torch.save(model, os.path.join(SAVE_PT_DIR,'last2.pt'))
    print("model saved at weights")

    for y in range(0, len(train_loss)):
        if (y + 1) % inter_val == 0:
            train_acc_2.append(train_acc[y])
            train_loss_2.append(train_loss[y])
    plt.figure(figsize=(22, 8))

    x1_train = range(0, len(train_acc_2))
    x2_train = range(0, len(train_loss_2))
    y1_train = train_acc_2
    y2_train = train_loss_2

    x1_val = range(0, len(valid_loss))
    x2_val = range(0, len(valid_acc))
    y1_val = valid_loss
    y2_val = valid_acc

    plt.subplot(2, 1, 1)
    plt.plot(x2_train, y2_train, '-', color='blue', label='Train Loss')
    plt.plot(x1_val, y1_val, '-', color='red', label='Val Loss')
    plt.title('Training and Validation Accuracy')
    plt.ylabel('Accurac5555y')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(x1_train, y1_train, '-', color='blue', label='Train Acc')
    plt.plot(x2_val, y2_val, '-', color='red', label='Val Acc')
    plt.title('Training and Validation Acc')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()

    plt.show()


def val_model(model, valid_loader, criterion, device):
    valid_epoch_loss = 0.0
    valid_epoch_acc = 0.0

    model.eval()
    valid_bar = tqdm(valid_loader)
    for i, (img, label) in enumerate(valid_bar):
        img = img.to(device)
        label = label.to(device)

        B, V, C, H, W = img.size()
        img = img.view(-1, C, H, W)
        img = img.to(device, non_blocking=True)
        # b,v = label.shape
        # label=label.view(b*v)

        with torch.no_grad():
            output, _ = model(img)
        output = (output[1] + output[0])

        loss = criterion(output, label)
        valid_epoch_loss += loss.item()
        valid_epoch_acc += (output.argmax(dim=1) == label).sum()

    val_acc_mean = valid_epoch_acc / (len(valid_dataset) * NUM_VIEW)
    val_loss_mean = valid_epoch_loss / len(valid_loader)

    print('valid loss: {:.3f} valid acc: {:.3f}'.format(val_loss_mean, val_acc_mean*2))
    return val_acc_mean, val_loss_mean


if __name__ == '__main__':
    # Best performance observed with seed = 1001 on our setup (Ubuntu + RTX 3090/4090).
    # Slight variations may occur on other platforms such as MacBook M3 or Windows.
    seed_everything(1001)
    # please run the preprocess/DataEnhance.py first before running this file
    # -----your path-----
    DATA_PATH = "/Datasets/aptos2019"
    TRAIN_PATH = "train_preprocessed"
    VAL_PATH = "val_preprocessed"
    TEST_PATH = "test_preprocessed"
    # SAVE_IMG_DIR = 'imgs'
    SAVE_PT_DIR = 'weights'
    NUM_VIEW = 2
    IMAGE_SIZE = 224
    LR = 0.00001
    N_EPOCHS = 100
    DEPTH = 12
    HEAD = 9
    BATCH_SIZE = 14

    train_csv_path = os.path.join(DATA_PATH, 'train2view.csv')
    assert os.path.exists(train_csv_path), '{} path is not exists...'.format(train_csv_path)
    val_csv_path = os.path.join(DATA_PATH, 'val2view.csv')
    val_df = pd.read_csv(val_csv_path)

    all_data = pd.read_csv(train_csv_path)
    all_data.head()

    transform_train = transform.Compose([
        transform.ToPILImage(),
        transform.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transform.RandomHorizontalFlip(p=0.3),
        transform.RandomVerticalFlip(p=0.3),
        transform.ToTensor(),
        # transform.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # transform.Normalize((0.28533897, 0.29217866, 0.29960716), (0.13518676, 0.13925494, 0.13853589)),
    ])

    transform_test = transform.Compose([
        transform.ToPILImage(),
        transform.Resize((224, 224)),
        transform.ToTensor(),
        # transform.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_dataset = TwoviewImgDataset(TRAIN_PATH, all_data, transform=transform_train)
    valid_dataset = TwoviewImgDataset(VAL_PATH, val_df, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=False)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=False)
    print(len(train_dataset), len(valid_dataset))

    main(model_name=f'main_{LR}_{N_EPOCHS}_d{DEPTH}_h{HEAD}_CCSKfomer', N_EPOCHS=N_EPOCHS, LR=LR, depth=DEPTH, head=HEAD)
