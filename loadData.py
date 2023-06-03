import os
import collections
from params import synonyms
import random

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
            L = map(lambda x: abs(x), L)
            labels += L    
    
    # Remove labels that only contain one review
    counter = collections.Counter(labels)
    for c in counter.items():
        if c[1] == 1:
            for i, v in enumerate(labels):
                if v == c[0]:
                    labels.remove(v)
                    reviews.remove(reviews[i])
    
    return reviews, labels

def dataAugmention(reviews, labels):
    augment_reviews, augment_labels = [], []
    for review, label in zip(reviews, labels):
        for synonym in synonyms:
            for syn in synonym:
                index = review.find(syn)
                if index != -1:
                    for s in synonym:
                        if s != syn:
                            new_review = review[0:index]+s+review[index+len(syn):]
                            augment_reviews.append(new_review)
                            augment_labels.append(label)

    reviews += augment_reviews
    labels += augment_labels
    
    return reviews, labels

def mergeTextLabels(reviews, labels):
    with open('./data/data_for_augment/before.txt', 'w+', encoding='utf-8') as file:
        for review, label in zip(reviews, labels):
            row = str(label)+'\t'+review+'\n'
            file.write(row)
        
    
if __name__ == '__main__':
    reviews, labels = readData()
    mergeTextLabels(reviews, labels)