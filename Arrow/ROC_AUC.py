import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc

def draw_ROC_Curve(ac, pro):
    uac = ['NG', 'OK']
    upro = sorted(set(pro))
    x, y = [], []
    M = 0
    for i in upro:
        tp, fn, fp, tn = 0, 0, 0, 0
        for j in range(len(pro)):
            if pro[j] >= i and ac[j] == uac[0]:
                tp += 1
            elif pro[j] < i and ac[j] == uac[0]:
                fn += 1
            elif pro[j] >= i and ac[j] == uac[1]:
                fp += 1
            elif pro[j] < i and ac[j] == uac[1]:
                tn += 1

        x.append(fp / (fp + tn))
        y.append(tp / (tp + fn))
        #print(fp / (fp + tn), tp / (tp + fn))
        if tp / (tp + fn) - fp / (fp + tn) >= M:
            M = tp / (tp + fn) - fp / (fp + tn)
            MM = i
    plt.plot(x, y)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    print(f'threshold: {MM}\nauc: {auc(x,y)}')
    plt.show()


a = pd.read_csv('C:/momentum/train.csv')
for i in range(10):
    b = pd.read_csv(f'C:/momentum/save{i}.csv')
    for j in range(len(a)):
        if a['label'][j] != b['label'][j]:
            print(i, j)
# for i in range(len(a)):
#     if a['label'][i] == '10-1':
#         a['label'][i] = '10'
#     elif a['label'][i] == '10-2':
#         a['label'][i] = '0'

#a.to_csv('C:/momentum/train.csv', index=False)
