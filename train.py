from numpy import argmax
from pathlib import Path
import torch
from tqdm import tqdm
import torchvision.utils as tvutils

from cnn_finetune import make_model
from training_datasetting import data_setting
from training_dataloader import *
from utils import *
from models import *
from sklearn.metrics import roc_curve, precision_recall_curve, auc, f1_score, roc_auc_score
from sklearn.model_selection import KFold
import pandas as pd
import csv
from numpy import argmax


print('== Data Setting ==')
data_train, data_test, norm_train, abnorm_train = np.array(data_setting())


print('== Environment Setting ==')
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('CUDA available:', torch.cuda.is_available())

NUM_CLASSES = 1
MODEL_NAME = 'resnet18'

for _ in range(10):
    f = open('result.csv', 'w', encoding='utf-8', newline='')
    wr = csv.writer(f)
    wr.writerow(['val_precision', 'val_sensitivity', 'val_specificity', 'val_auc', 'val_accuracy'])
    f.close()

    acc_list, auc_list, pre_list, sen_list, spe_list, f1_list = [],[],[],[],[],[]


    model = make_model(MODEL_NAME, num_classes=NUM_CLASSES, pretrained=False)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    model.to(device)

    bce_criterion = torch.nn.BCELoss()
    mse_criterion = torch.nn.MSELoss()
    distance_criterion = torch.jit.script(Distance_euclidean())
    # criterion = torch.nn.BCEWithLogitsLoss()
    learning_rate = 2e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    print('== Data Loadding ==')
    BATCH_SIZE = 24
    train_dataset = ImageDataset_Train(data_train, norm_train, abnorm_train, is_train=True, )
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = ImageDataset(data_test, is_train=False)
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)

    NUM_EPOCH = 120
    best_acc, best_auc = 0.0, 0.0

    print('== Start Training ==')

    for epoch in range(NUM_EPOCH):
        # print('Epoch {}/{}'.format(epoch, NUM_EPOCH - 1))
        running_train_loss = 0.0
        running_class_loss = 0.0
        running_contra_loss = 0.0
        running_train_accuracy = 0.0
        train_total = 0

        model.train()        
        for i, (imgs, train_labels, nega_imgs, train_nega_labels, pos_imgs, train_pos_labels) in enumerate(train_dataloader):
            optimizer.zero_grad()
            output = 0
            features_list, nega_features_list, pos_features_list = [], [], []
            

            for step, img in enumerate(imgs):
                img, train_labels = img.to(device), train_labels.to(device)
                nega_img, train_nega_labels = nega_imgs[step].to(device), train_nega_labels.to(device)
                pos_img, train_pos_labels = pos_imgs[step].to(device), train_pos_labels.to(device)

                features_list.append(model(img))
                nega_features_list.append(model(nega_img).detach())
                pos_features_list.append(model(pos_img).detach())

                output += torch.sigmoid(features_list[step])
            outputs = output / 4
            train_preds = outputs >= torch.FloatTensor([0.5]).to(device)
            running_train_accuracy += (train_preds == train_labels).sum().item()

            bce_loss = bce_criterion(outputs, train_labels.float())

            negative_loss, positive_loss = 0, 0
            for k in range(len(features_list)):
                negative_loss += distance_criterion(features_list[k], nega_features_list[k])
                positive_loss += distance_criterion(features_list[k], pos_features_list[k])

            contrastive_loss = torch.relu(positive_loss - negative_loss).mean()
            
            train_loss = bce_loss + contrastive_loss

            running_train_loss += train_loss.item()
            running_class_loss += bce_loss.item()
            running_contra_loss += contrastive_loss.item()

            train_total += train_labels.size(0)

            train_loss.backward()
            optimizer.step()

            # if num_kf == 0 and epoch == 0 and i == 0:
            #     Path('./sample_imgs/').mkdir(parents=True, exist_ok=True)
            #     save_imgs = torch.cat([anchor_imgs, negative_imgs], 2)
            #     tvutils.save_image(save_imgs, './sample_imgs/training_imgs.jpg')
            #     print('train img save')

        
        train_loss_value = running_train_loss/len(train_dataloader)
        classification_loss_value = running_class_loss/len(train_dataloader)
        contrastive_loss_value = running_contra_loss/len(train_dataloader)
        train_accuracy = (100 * running_train_accuracy / train_total)
        
        
        all_labels = torch.tensor([]).to(device)
        all_preds_prob = torch.tensor([]).to(device)

        with torch.no_grad():
            model.eval()
            for i, (imgs, val_labels) in enumerate(val_dataloader):

                outputs = 0
                for step, img in enumerate(imgs):
                    img, val_labels = img.to(device), val_labels.to(device)

                    outputs += torch.sigmoid(model(img))
                outputs = outputs / 4             

                all_labels = torch.cat([all_labels, val_labels], dim=0)
                all_preds_prob = torch.cat([all_preds_prob, outputs], dim=0)

            all_labels, all_preds_prob = all_labels.cpu(), all_preds_prob.cpu()

            fpr, tpr, thr = roc_curve(all_labels, all_preds_prob)            
            val_auc = auc(fpr, tpr)

            best_running_pre, best_running_sen, best_running_spe, best_running_acc = 0, 0, 0, 0   
            j = tpr - fpr
            ix = argmax(j)        
            best_thr = thr[ix]

            best_all_preds = (all_preds_prob >= best_thr)
            best_running_pre, best_running_sen, best_running_spe, best_running_acc = evaluation_per_binayclass(all_labels, best_all_preds)
            
            if val_auc >= best_auc:
                best_acc = best_running_acc
                best_auc = val_auc
                best_result = [best_running_pre, best_running_sen, best_running_spe, best_auc, best_running_acc]
                filename = '{}'.format(str(best_auc))
                model_ckpt_save(model, filename)
                binary_confusion_matrix(all_labels, best_all_preds, 0)
                plot_binaryclass_roc_curve(fpr, tpr, val_auc, filename +'_roc_auc_curve.png')
            
        print('Model:', MODEL_NAME, ", Epoch: [%d/%d]" % (epoch+1, NUM_EPOCH), 
        ', T- Loss: %.4f' % (train_loss_value), ', T- Accuracy: %.4f %%' % train_accuracy,
        ', V- Precision: %.4f' % best_running_pre,', V- Sensitivity: %.4f %%' % best_running_sen,
        ', V- Specificity: %.4f' % best_running_spe,', V- Accuracy: %.4f %%' % best_running_acc,
        ', V- AUC: %.4f %%' % val_auc,
        ', Best AUC: %.4f %%' % best_auc)

    acc_list.append(best_acc)
    with open('result.csv', 'a', newline='') as csvfile:
        wr = csv.writer(csvfile)
        wr.writerow(best_result)

    print(acc_list, sum(acc_list)/len(acc_list))
