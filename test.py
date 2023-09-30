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

acc_list, auc_list, pre_list, sen_list, spe_list, f1_list = [],[],[],[],[],[]
model = make_model(MODEL_NAME, num_classes=NUM_CLASSES, pretrained=False)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = torch.nn.DataParallel(model)

pre_trained_path_list = ['/home/compu/working/breast_project/code/implement/paper/6/0.1>=/models_without/0.7416368799614053-epo.ckpt',
'/home/compu/working/breast_project/code/implement/paper/6/0.1>=/models_without/0.7504193895004605-epo.ckpt',
'/home/compu/working/breast_project/code/implement/paper/6/0.1>=/models_without/0.7618361036796631-epo.ckpt',
'/home/compu/working/breast_project/code/implement/paper/6/0.1>=/models_without/0.7712435858076401-epo.ckpt',
'/home/compu/working/breast_project/code/implement/paper/6/0.1>=/models_without/0.7811855839656155-epo.ckpt',]

label_name_list = ['1', '2', '3', '4', '5']

all_fpr_list, all_tpr_list, all_auc_list = [], [], []
sen_box, spe_box, auc_box, acc_box = [], [], [], []

for no in range(len(pre_trained_path_list)):
    pre_trained_path = pre_trained_path_list[no]
    model.load_state_dict(torch.load(pre_trained_path))
    model.to(device)

    bce_criterion = torch.nn.BCELoss()
    mse_criterion = torch.nn.MSELoss()
    distance_criterion = torch.jit.script(Distance_euclidean())

    train_dataset = ImageDataset(data_train, is_train=False)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)

    val_dataset = ImageDataset(data_test, is_train=False)
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)

    NUM_EPOCH = 120
    best_acc, best_auc = 0.0, 0.0

    print('== Start Training ==')

    # all_ids = []
    # all_labels = torch.tensor([]).to(device)
    # all_preds_prob = torch.tensor([]).to(device)


    # train_csvfile_name = 'train_result.csv'
    # f = open(train_csvfile_name, 'w', encoding='utf-8', newline='')
    # wr = csv.writer(f)
    # wr.writerow(['id', 'label', 'prediction_probability', 'prediction_result'])
    # f.close()


    # with torch.no_grad():
    #     model.eval()
    #     for i, (imgs, val_labels, val_ids) in enumerate(train_dataloader):

    #         outputs = 0
    #         for step, img in enumerate(imgs):
    #             img, val_labels = img.to(device), val_labels.to(device)

    #             outputs += torch.sigmoid(model(img))
    #         outputs = outputs / 4             

    #         all_ids.append(val_ids)
    #         all_labels = torch.cat([all_labels, val_labels], dim=0)
    #         all_preds_prob = torch.cat([all_preds_prob, outputs], dim=0)

    #     all_labels, all_preds_prob = all_labels.cpu(), all_preds_prob.cpu()

    #     fpr, tpr, thr = roc_curve(all_labels, all_preds_prob)            
    #     val_auc = auc(fpr, tpr)

    #     best_running_pre, best_running_sen, best_running_spe, best_running_acc = 0, 0, 0, 0   
    #     j = tpr - fpr
    #     ix = argmax(j)        
    #     best_thr = thr[ix]
    #     # best_thr = 0.5
    #     best_all_preds = (all_preds_prob >= best_thr)
    #     best_running_pre, best_running_sen, best_running_spe, best_running_acc = evaluation_per_binayclass(all_labels, best_all_preds)

    # with open(train_csvfile_name, 'a', newline='') as csvfile:
    #     wr = csv.writer(csvfile)

    #     for i in range(len(all_ids)):
    #         wr.writerow([all_ids[i][0], all_labels[i][0].item(), all_preds_prob[i][0].item(), best_all_preds[i][0].item()])


    # print('Model:', MODEL_NAME, ', Threshold:', best_thr,
    # ', V- Sensitivity: %.4f %%' % best_running_sen, ', V- Specificity: %.4f' % best_running_spe,
    # ', V- AUC: %.4f %%' % val_auc,', V- Accuracy: %.4f %%' % best_running_acc)




    all_ids = []
    all_labels = torch.tensor([]).to(device)
    all_preds_prob = torch.tensor([]).to(device)

    csvfile_name = 'validation_result.csv'
    f = open(csvfile_name, 'w', encoding='utf-8', newline='')
    wr = csv.writer(f)
    wr.writerow(['id', 'label', 'prediction_probability', 'prediction_result'])
    f.close()



    with torch.no_grad():
        model.eval()
        for i, (imgs, val_labels, val_ids) in enumerate(val_dataloader):

            outputs = 0
            for step, img in enumerate(imgs):
                img, val_labels = img.to(device), val_labels.to(device)
                result = model(img)
                print('result', result)
                outputs += torch.sigmoid(result)
            outputs = outputs / 4             

            all_ids.append(val_ids)
            all_labels = torch.cat([all_labels, val_labels], dim=0)
            all_preds_prob = torch.cat([all_preds_prob, outputs], dim=0)

        all_labels, all_preds_prob = all_labels.cpu(), all_preds_prob.cpu()

        fpr, tpr, thr = roc_curve(all_labels, all_preds_prob)            
        val_auc = auc(fpr, tpr)

        all_fpr_list.append(fpr)
        all_tpr_list.append(tpr)
        all_auc_list.append(val_auc)
        
        filename = pre_trained_path.split('/')[-1][:6]
        plot_binaryclass_roc_curve(fpr, tpr, val_auc, filename +'_roc_auc_curve.png')

        best_running_pre, best_running_sen, best_running_spe, best_running_acc = 0, 0, 0, 0   
        j = tpr - fpr
        ix = argmax(j)        
        best_thr = thr[ix]
        # best_thr = 0.5
        best_all_preds = (all_preds_prob >= best_thr)
        best_running_pre, best_running_sen, best_running_spe, best_running_acc = evaluation_per_binayclass(all_labels, best_all_preds)

    sen_box.append(best_running_sen)
    spe_box.append(best_running_spe)
    auc_box.append(val_auc)
    acc_box.append(best_running_acc)

    with open(csvfile_name, 'a', newline='') as csvfile:
        wr = csv.writer(csvfile)

        for i in range(len(all_ids)):
            wr.writerow([all_ids[i][0], all_labels[i][0].item(), all_preds_prob[i][0].item(), best_all_preds[i][0].item()])


    print('Model:', MODEL_NAME, ', Threshold:', best_thr,
    ', V- Sensitivity: %.4f %%' % best_running_sen, ', V- Specificity: %.4f' % best_running_spe,
    ', V- AUC: %.4f %%' % val_auc,', V- Accuracy: %.4f %%' % best_running_acc)




save_path = './roc_auc_curve/'
Path(save_path).mkdir(parents=True, exist_ok=True)

lw = 2
color = ['orange', 'red', 'green', 'blue', 'yellow']
for no in range(len(all_auc_list)):
    plt.plot(all_fpr_list[no], all_tpr_list[no], color=color[no], lw=lw, label= label_name_list[no] +' ROC curve (area = %.4f)' % all_auc_list[no])

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("Receiver operating characteristic")
plt.legend(loc='lower right')
plt.savefig(save_path + 'all_roc_auc_curve.png')
plt.close()

print('sen box:', sen_box)
mean_sen, lower_sen, upper_sen = confidenc_intervals(sen_box)
print('spe_box:', spe_box)
mean_spe, lower_spe, upper_spe = confidenc_intervals(spe_box)
print('auc_box:', auc_box)
mean_auc, lower_auc, upper_auc = confidenc_intervals(auc_box)
print('acc_box:', acc_box)
mean_acc, lower_acc, upper_acc = confidenc_intervals(acc_box)
print('sen mean, lower, upper:', mean_sen, lower_sen, upper_sen)
print('spe mean, lower, upper:', mean_spe, lower_spe, upper_spe)
print('auc mean, lower, upper:', mean_auc, lower_auc, upper_auc)
print('acc mean, lower, upper:', mean_acc, lower_acc, upper_acc)