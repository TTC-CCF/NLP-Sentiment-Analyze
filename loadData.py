import os

reviews_file_path = './data/reviews'
labels_file_path = './data/labels'
def readData():
    reviews_files = os.listdir(reviews_file_path)
    labels_files = os.listdir(labels_file_path)
    reviews = []
    labels = []
    for file in reviews_files:
        with open(os.path.join(reviews_file_path, file), 'r', encoding='utf-8') as f:
            text = f.read()
            text = text.split('\n')
            text = [word for word in text if word != '']
            reviews += text
    for file in labels_files:
        with open(os.path.join(labels_file_path, file), 'r', encoding='utf-8') as f:
            L = f.read()
            L = L.split('\n')
            L = [int(lab) for lab in L if lab != '']
            labels += L
    
    return reviews, labels
    
if __name__ == '__main__':
    print(readData())