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
            L = map(lambda x: abs(x)+30 if x < 0 else x, L)
            labels += L
    
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
    counter = collections.Counter(labels)
    for c in counter.items():
        if c[1] == 1:
            for i, v in enumerate(labels):
                if v == c[0]:
                    labels.remove(v)
                    reviews.remove(reviews[i])
    
    return reviews, labels
    
if __name__ == '__main__':
    readData()