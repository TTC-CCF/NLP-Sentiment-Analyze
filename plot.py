import matplotlib.pyplot as plt
import numpy as np

def plotAccuracy(title, arr1, arr2, file):
    arr1 = np.load(arr1)
    arr2 = np.load(arr2)

    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot([i for i in range(len(arr1))], arr1, label='aug')
    plt.plot([i for i in range(len(arr2))], arr2, label='unaug')
    plt.legend()
    plt.plot()
    plt.savefig(file)
    plt.show()
    plt.close()
    
if __name__ == '__main__':
    loss1 = './loss/aug_teams.npy'
    loss2 = './loss/unaug_teams.npy'
    
    loss3 = './loss/aug_sent.npy'
    loss4 = './loss/unaug_sent.npy'
    
    accu1 = './accu/aug_teams.npy'
    accu2 = './accu/unaug_teams.npy'
    
    accu3 = './accu/aug_sent.npy'
    accu4 = './accu/unaug_sent.npy'
    
    plotAccuracy('Teams classification loss', loss1, loss2, './plot/loss_teams.png')
    plotAccuracy('Sentimental analysis loss', loss3, loss4, './plot/loss_sent.png')

    plotAccuracy('Teams classification Accuracy', accu1, accu2, './plot/accu_teams.png')
    plotAccuracy('Sentimental analysis Accuracy', accu3, accu4, './plot/accu_sent.png')    
    