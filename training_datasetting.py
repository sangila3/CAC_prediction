import math
import csv
from sklearn.model_selection import train_test_split
import random
import pandas as pd
import os
import numpy as np

def data_setting():
    csv_data_tdf = pd.read_csv('/home/compu/working/breast_project/preprocessing/binary_test_with_age_menopause.csv')
    tr_id_list = csv_data_tdf.to_numpy().tolist()

    count = 0
    normList, abnormList= [], []


    for info in tr_id_list:
        if int(info[1]) == 0:
            normList.append([info[0], 0])
        else:
            abnormList.append([info[0], 1])

    
    print(len(normList), len(abnormList)) # 3412 3031
    # norm_list, norm_ex_data = normList[:len(abnormList)], normList[len(abnormList):]
    norm_list = normList
    ab_list = abnormList

            
   
    print(len(norm_list), len(ab_list)) # 3412 3031


    n_train = int(int(len(ab_list) / 4) * 0.2) * 4

    test_normlist, test_abnormlist = norm_list[:n_train], ab_list[:n_train]
    train_normlist, train_abnormlist = norm_list[n_train:], ab_list[n_train:]

    train_list = train_normlist + train_abnormlist
    test_list = test_normlist + test_abnormlist   


    return np.array(train_list), np.array(test_list), np.array(train_normlist), np.array(train_abnormlist)

data_setting()