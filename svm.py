from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer ,CountVectorizer
from loadData import readData, dataAugmention
from transformers import AutoModel, AutoTokenizer, logging
import torch
import pickle
import jieba
from params import LabeltoTeamsDict, model_name, BATCH_SIZE

logging.set_verbosity_error()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
feature_model = AutoModel.from_pretrained('bert-base-chinese')
tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
feature_model.to(device)

def extractFeatures(texts):
    encoded = tokenizer(texts, padding='max_length', truncation=True, max_length=100, return_tensors='pt')
    encoded.to(device)
    with torch.no_grad():
        features = []
        for step in range(0, len(texts), BATCH_SIZE):
            input_ids = encoded['input_ids'][step: step+BATCH_SIZE]
            attention_mask = encoded['attention_mask'][step: step+BATCH_SIZE]
            outputs = feature_model(input_ids, attention_mask).last_hidden_state
            outputs = outputs.detach().cpu().numpy().reshape(outputs.shape[0], -1)
            for output in outputs:
                features.append(output)
    print(f'Extracted {len(features)} features')
    return features

def encodeTexts(texts):
    return tokenizer(texts, padding='max_length', truncation=True, max_length=100, return_tensors='pt')['input_ids']


if __name__ == '__main__':
    # reviews, labels = readData()
    # reviews, labels = dataAugmention(reviews, labels)
    reviews, labels = [], []
    with open('./data/train_augmented.txt', 'r', encoding='utf-8') as file:
        for line in file.readlines():
            l, r = line.split('\t')
            labels.append(int(l))
            reviews.append(r)
    # reviews, labels = dataAugmention(reviews, labels)
    
    vectorizer = TfidfVectorizer()
    print('vectorizing')
    X = vectorizer.fit_transform(reviews).toarray()
    train_texts, test_texts, train_labels, test_labels = train_test_split(X, labels, stratify=labels)
    
    # print('training')
    # clf = svm.SVC(kernel='linear')
    # clf.fit(train_texts, train_labels)
    # pickle.dump(clf, open('results/trained_svm_model.bin', 'wb'))

    print('predicting')
    loaded_clf = pickle.load(open('results/trained_svm_model.bin', 'rb'))
    # pred = loaded_clf.predict(test_texts)
    # back_testing = loaded_clf.predict(train_texts)
    # print(classification_report(test_labels, pred))
    new_test = ['金塊 冠軍 ！','金塊','湖人 很穩', '阿肥衝阿!', '勇士衛冕很穩', '咖哩是要全隊花心思去防守的，不可能只有范德標']
    nt = []
    for t in new_test:
        seg = ' '.join(jieba.cut(t))
        nt.append(seg)
    vectored = vectorizer.transform(nt).toarray()
    print([LabeltoTeamsDict[i] for i in loaded_clf.predict(vectored)])
    # train_features = extractFeatures(train_texts)
    # test_features = extractFeatures(test_texts)
    
    # train_features = encodeTexts(train_texts)
    # test_features = encodeTexts(test_texts)
    # clf = svm.SVC(kernel='linear')
    # clf.fit(train_features, train_labels)
    # pickle.dump(clf, open('results/trained_svm_model.bin', 'wb'))
    
    # loaded_clf = pickle.load(open('results/trained_svm_model.bin', 'rb'))
    # pred = loaded_clf.predict(test_features)
    # print(classification_report(test_labels, pred))
    
    # new_text = ['太陽最近很強喔', '湖人要贏了吧，金塊要怎麼贏?', '綠seafood要出招了']
    # inference_features = encodeTexts(new_text)
    # infer = loaded_clf.predict(inference_features)
    # print([LabeltoTeamsDict[i] for i in infer])