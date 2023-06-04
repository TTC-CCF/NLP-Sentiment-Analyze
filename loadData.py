import os
import collections
import copy
from eda import eda



reviews_file_path = './data/reviews'
labels_file_path = './data/labels'
def readData():
    reviews_files = os.listdir(reviews_file_path)
    labels_files = os.listdir(labels_file_path)
    reviews = []
    teams_labels = []
    sent_labels = []
    for review_file, label_file in zip(reviews_files, labels_files):
        with open(os.path.join(reviews_file_path, review_file), 'r', encoding='utf-8') as f:
            text = f.read()
            text = text.split('\n')
            text = [word for word in text if word != '']
            reviews += text
        with open(os.path.join(labels_file_path, label_file), 'r', encoding='utf-8') as f:
            L = f.read()
            L = L.split('\n')
            L = [int(lab) for lab in L if lab != '']
            teams = list(map(lambda x: abs(x), L))
            sent = list(map(lambda x: 1 if x > 0 else (2 if x < 0 else 0), L))
            
            teams_labels += teams
            sent_labels += sent
            
        print(f'{review_file} contains {len(text)} reviews, {len(teams)} labels')
        
            
    
    # Remove labels that only contain one review
    temp = copy.deepcopy(reviews)
    teams_reviews, sent_reviews = [], []
    teams_counter = collections.Counter(teams_labels)
    sent_counter = collections.Counter(sent_labels)
    for c in teams_counter.items():
        if c[1] == 1:
            for i, v in enumerate(teams_labels):
                if v == c[0]:
                    teams_labels.remove(v)
                    temp.remove(temp[i])
    teams_reviews = copy.deepcopy(temp)
    temp = copy.deepcopy(reviews)
    for c in sent_counter.items():
        if c[1] == 1:
            for i, v in enumerate(sent_labels):
                if v == c[0]:
                    sent_labels.remove(v)
                    temp.remove(temp[i])
    sent_reviews = copy.deepcopy(temp)
    
    return teams_reviews, sent_reviews, teams_labels, sent_labels

def dataAugmentation(reviews, labels):
    new_reviews, new_labels = [], []
    for r, l in zip(reviews, labels):
        augmented_reviews = eda(r, num_aug=10)
        for ar in augmented_reviews:
            new_reviews.append(ar)
            new_labels.append(l)
    return new_reviews, new_labels

def loadValidate():
    validate_reviews_path = './data/validate/reviews'
    validate_labels_path = './data/validate/labels'
    reviews_files = os.listdir(validate_reviews_path)
    labels_files = os.listdir(validate_labels_path)
    reviews, teams_labels, sent_labels = [], [], []
            
    for review_file, label_file in zip(reviews_files, labels_files):
        with open(os.path.join(validate_reviews_path, review_file), 'r', encoding='utf-8') as f:
            text = f.read()
            text = text.split('\n')
            text = [word for word in text if word != '']
            reviews += text
        with open(os.path.join(validate_labels_path, label_file), 'r', encoding='utf-8') as f:
            L = f.read()
            L = L.split('\n')
            L = [int(lab) for lab in L if lab != '']
            teams = list(map(lambda x: abs(x), L))
            sent = list(map(lambda x: 1 if x > 0 else (2 if x < 0 else 0), L))
            
            teams_labels += teams
            sent_labels += sent
    return reviews, teams_labels, sent_labels


def showLabelsFreq(labels):
    n = len(labels)
    l_dict = collections.Counter(labels)
    sum = 0.0
    for l, freq in l_dict.items():
        print(f'{l}      {"{:.2f}".format(freq*100/n)}%')
        
    
if __name__ == '__main__':
    teams_reviews, sent_reviews, teams_labels, sent_labels = readData()
    augmented_reviews, augmented_labels = dataAugmentation(teams_reviews, teams_labels)
    showLabelsFreq(augmented_labels)
    augmented_reviews, augmented_labels = dataAugmentation(sent_reviews, sent_labels)
    
    showLabelsFreq(augmented_labels)
