import matplotlib.pyplot as plt
import os
import itertools
import numpy as np
def plot_loss(df,log_path):
    plt.plot(df['epoch'].values,df['loss_value'].values, marker='o', label='Train Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()  # 显示图例
    plt.grid(True)
    plt.savefig(os.path.join(log_path,"loss_curves.png"), dpi=300, bbox_inches='tight')  # 保存为 PNG 格式
    # 显示图表
    plt.show()

def plot_acc(df,best_epoch,log_path):

    plt.plot(df['epoch'].values, df['val_acc'].values, marker='o', label='Validation Acc', color='orange')
    plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch:{best_epoch}')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.title('Val set Accuracy Curves')
    plt.legend()  # 显示图例
    plt.grid(True)
    plt.savefig(os.path.join(log_path,"Val_set_Accuracy_Curves.png"), dpi=300, bbox_inches='tight')  # 保存为 PNG 格式
    # 显示图表
    plt.show()


# 绘制混淆矩阵
def plot_confusion_matrix(cm1, ax,classes,title,cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    cm = cm1.astype('float') / cm1.sum(axis=1)[:, np.newaxis]
    ax.imshow(cm1, interpolation='nearest', cmap=cmap)
    ax.set_title(title, fontproperties='Times New Roman',fontsize=20,weight='bold')
    #ax.colorbar()
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks, classes,fontsize=12,fontproperties='Times New Roman',weight='bold')
    ax.set_yticks(tick_marks, classes,fontsize=12,fontproperties='Times New Roman',weight='bold')
    thresh = cm.max() / 2.
    cm2=np.ones(len(classes)*len(classes)).reshape(len(classes),len(classes))
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        cm2[i,j]=100*cm[i,j]
        cm2[i,j]=float('%.1f'%cm2[i,j])
        str1=str(cm1[i,j])+'\n'+'('+str(cm2[i,j])+'%)'
        ax.text(j, i, str1,
                 horizontalalignment="center",
                 verticalalignment='center',
                 color="white" if cm[i, j] > thresh else "black")
    #ax.tight_layout()
    ax.set_ylabel('True label',fontsize=16,fontproperties='Times New Roman',weight='bold')
    ax.set_xlabel('Predicted label',fontsize=16,fontproperties='Times New Roman',weight='bold')
    #ax.gcf().subplots_adjust(left=0.05, top=0.99, bottom=0.19, right=None)
    #ax.rcParams['figure.figsize'] = (12.0,8.0)

def plot_multilabel_cf(cnf_matrix,test_acc,log_path):
    attack_types = ['0', '1']
    title = ['PE', 'PC', 'PS', 'PVC', 'PP']
    for i in range(5):
        plt.subplot(2, 3, i + 1)
        ax = plt.gca()
        plot_confusion_matrix(cnf_matrix[i], ax,title=title[i], classes=attack_types)
    plt.tight_layout()
    plt.savefig(os.path.join(log_path, f"test_set_acc={test_acc}.png"), dpi=300, bbox_inches='tight')
    plt.show()




