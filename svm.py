from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from loadData import readData
from transformers import AutoModel, AutoTokenizer
import torch
import pickle
from params import LabeltoTeamsDict

model_name = 'xlm-roberta-base'
feature_model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

reviews, labels = readData()
                
encoded = tokenizer(reviews, padding='max_length', truncation=True, max_length=100, return_tensors='pt')

with torch.no_grad():
    output = feature_model(**encoded)
    features = output.last_hidden_state


train_texts, test_texts, train_labels, test_labels = train_test_split(features, labels, stratify=labels)

train_texts = train_texts.reshape(train_texts.shape[0], -1)
test_texts = test_texts.reshape(test_texts.shape[0], -1)


clf = svm.SVC(kernel='linear')
clf.fit(train_texts, train_labels)
pickle.dump(clf, open('results/trained_svm_model.bin', 'wb'))


loaded_clf = pickle.load(open('results/trained_svm_model.bin', 'rb'))
pred = loaded_clf.predict(test_texts)
print(classification_report(test_labels, pred))

new_text = ['Curry要發威了嗎?', '湖人要贏了吧，金塊要怎麼贏?', '綠seafood要出招了']
with torch.no_grad():
    feature = feature_model(**(tokenizer(new_text, padding='max_length', truncation=True, max_length=100, return_tensors='pt'))).last_hidden_state
feature = feature.reshape(feature.shape[0], -1)
for p in loaded_clf.predict(feature):
    print(LabeltoTeamsDict[p])