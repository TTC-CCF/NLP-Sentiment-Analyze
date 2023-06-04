from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from loadData import readData, dataAugmentation, loadValidate
import pickle
import jieba
import os
from params import LabeltoTeamsDict, BATCH_SIZE, LabeltoSentDict


def preProcessing(reviews):
    processed = []
    for r in reviews:
        seg = ' '.join(jieba.cut(r))
        processed.append(seg)
    
    return processed
        
if __name__ == '__main__':
    teams_reviews, sent_reviews, teams_labels, sent_labels = readData()
    v_reviews, v_teams_labels, v_sent_labels = loadValidate()
    
    teams = train_test_split(teams_reviews, teams_labels, stratify=teams_labels)
    sent = train_test_split(sent_reviews, sent_labels, stratify=sent_labels)
    
    test_teams_texts, test_teams_labels = preProcessing(teams[1]), teams[3]
    test_sent_texts, test_sent_labels = preProcessing(sent[1]), sent[3]
    v_reviews = preProcessing(v_reviews)
    
    teams_save_path = 'results/trained_svm_model_teams.bin'
    sent_save_path = 'results/trained_svm_model_sent.bin'
    teams_vectorizer_save_path = 'results/svm_trained_teams_vectorizer.bin'
    sent_vectorizer_save_path = 'results/svm_trained_sent_vectorizer.bin'
    
    # print('Augmenting train dataset...')
    # train_teams_texts, train_teams_labels = dataAugmentation(teams[0], teams[2])
    # train_sent_texts, train_sent_labels = dataAugmentation(sent[0], sent[2])
    # print(len(train_teams_texts))
    # print('Vectorizing...')
    # teams_vectorizer = TfidfVectorizer(max_features=2500)
    # V_train_teams_texts = teams_vectorizer.fit_transform(train_teams_texts).toarray()
    # V_test_teams_texts = teams_vectorizer.transform(test_teams_texts).toarray()
    # pickle.dump(teams_vectorizer, open(teams_vectorizer_save_path, 'wb'))
    
    # sent_vectorizer = TfidfVectorizer(max_features=2500)
    # V_train_sent_texts = sent_vectorizer.fit_transform(train_sent_texts).toarray()
    # pickle.dump(sent_vectorizer, open(sent_vectorizer_save_path, 'wb'))
    
    
    # teams_clf = svm.SVC(kernel='linear')
    # sent_clf = svm.SVC(kernel='linear')
    
    # print('Training Model for Teams Classification...')
    # teams_clf.fit(V_train_teams_texts, train_teams_labels)
    # pickle.dump(teams_clf, open(teams_save_path, 'wb'))

    # print('Training Model for Sentiment...')
    # sent_clf.fit(V_train_sent_texts, train_sent_labels)
    # pickle.dump(sent_clf, open(sent_save_path, 'wb'))
    

    print('Testing Teams Classification Model...')
    teams_loaded_vectorizer = pickle.load(open(teams_vectorizer_save_path, 'rb'))
    teams_loaded_clf = pickle.load(open(teams_save_path, 'rb'))
    
    V_test_teams_texts = teams_loaded_vectorizer.transform(test_teams_texts).toarray()
    teams_pred = teams_loaded_clf.predict(V_test_teams_texts)
    print(classification_report(test_teams_labels, teams_pred))
    
    print('Testing Sentiment Model...')
    sent_loaded_vectorizer = pickle.load(open(sent_vectorizer_save_path, 'rb'))
    sent_loaded_clf = pickle.load(open(sent_save_path, 'rb'))
    
    V_test_sent_texts = sent_loaded_vectorizer.transform(test_sent_texts).toarray()
    sent_pred = sent_loaded_clf.predict(V_test_sent_texts)
    print(classification_report(test_sent_labels, sent_pred))
    
    print('Testing on validating data...')
    v_teams_texts = teams_loaded_vectorizer.transform(v_reviews).toarray()
    v_sent_texts = sent_loaded_vectorizer.transform(v_reviews).toarray()
    teams_pred = teams_loaded_clf.predict(v_teams_texts)
    sent_pred = sent_loaded_clf.predict(v_sent_texts)
    print('Teams:', classification_report(v_teams_labels, teams_pred))
    print('Sentiment:', classification_report(v_sent_labels, sent_pred))

    
    # print('Predicting...')
    # new_test = ['金塊 冠軍 ！','金塊','湖人 很穩', '阿肥今天要加油欸', '勇士衛冕很穩', '咖哩是要全隊花心思去防守的，不可能只有范德標']
    
    # nt = preProcessing(new_test)
    # teams_vectorized = teams_loaded_vectorizer.transform(nt).toarray()
    # sent_vectorized = sent_loaded_vectorizer.transform(nt).toarray()
    
    # print([LabeltoTeamsDict[i] for i in teams_loaded_clf.predict(teams_vectorized)])
    # print([LabeltoSentDict[i] for i in sent_loaded_clf.predict(sent_vectorized)])