import matplotlib.pyplot as plt

def plotAccuracy(title, arr1, arr2, file):
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot([i for i in range(len(arr1))], arr1, label='title', color='gray')
    plt.plot([i for i in range(len(arr2))], arr2, label='title', color='gray')
    plt.plot()
    plt.savefig(f"./plot/{file}.png")
    plt.show()
    plt.close()
    
if __name__ == '__main__':
    loss1 = './loss/aug_teams.npy'
    loss2 = './loss/unaug_teams.npy'
    
    loss3 = './loss/aug_sent.npy'
    loss4 = './loss/aug_sent.npy'
    
    plotAccuracy('Teams classification loss', loss1, loss2, './plot/teams')
    plotAccuracy('Sentimental analysis loss', loss3, loss4, './plot/sent')
    
    