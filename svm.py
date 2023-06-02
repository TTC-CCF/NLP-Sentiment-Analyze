from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from loadData import readData
from transformers import AutoModel, AutoTokenizer, logging
import torch
from torch.utils.data import DataLoader
import pickle
from params import LabeltoTeamsDict, model_name, BATCH_SIZE

logging.set_verbosity_error()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
feature_model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
feature_model.to(device)

reviews, labels = readData()

encoded = tokenizer(reviews, padding='max_length', truncation=True, max_length=100, return_tensors='pt')
encoded.to(device)

with torch.no_grad():
    features = []
    for step in range(0, len(labels), BATCH_SIZE):
        input_ids = encoded['input_ids'][step: step+BATCH_SIZE]
        attention_mask = encoded['attention_mask'][step: step+BATCH_SIZE]
        outputs = feature_model(input_ids, attention_mask).last_hidden_state
        outputs = outputs.detach().cpu().numpy().reshape(outputs.shape[0], -1)
        for output in outputs:
            features.append(output)
        
train_texts, test_texts, train_labels, test_labels = train_test_split(features, labels, stratify=labels)


clf = svm.SVC(kernel='linear')
clf.fit(train_texts, train_labels)
pickle.dump(clf, open('results/trained_svm_model.bin', 'wb'))

loaded_clf = pickle.load(open('results/trained_svm_model.bin', 'rb'))
pred = loaded_clf.predict(test_texts)
print(classification_report(test_labels, pred))

new_text = ['太陽最近很強喔', '湖人要贏了吧，金塊要怎麼贏?', '綠seafood要出招了']
with torch.no_grad():
    encoded = tokenizer(new_text, padding='max_length', truncation=True, max_length=100, return_tensors='pt').to(device)
    feature = feature_model(**encoded).last_hidden_state
feature = feature.reshape(feature.shape[0], -1).detach().cpu().numpy()
for p in loaded_clf.predict(feature):
    print(LabeltoTeamsDict[p])