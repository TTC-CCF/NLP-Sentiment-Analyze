from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from loadData import readData, dataAugmention
from transformers import AutoModel, AutoTokenizer, logging
import torch
import pickle
from params import LabeltoTeamsDict, model_name, BATCH_SIZE

logging.set_verbosity_error()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
feature_model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
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


if __name__ == '__main__':
    reviews, labels = readData()
    train_texts, test_texts, train_labels, test_labels = train_test_split(reviews, labels, stratify=labels)
    train_texts, train_labels = dataAugmention(train_texts, train_labels)
    
    train_features = extractFeatures(train_texts)
    test_features = extractFeatures(test_texts)
    
    clf = svm.SVC(kernel='linear')
    clf.fit(train_features, train_labels)
    pickle.dump(clf, open('results/trained_svm_model.bin', 'wb'))
    
    loaded_clf = pickle.load(open('results/trained_svm_model.bin', 'rb'))
    pred = loaded_clf.predict(test_features)
    print(classification_report(test_labels, pred))
    
    new_text = ['太陽最近很強喔', '湖人要贏了吧，金塊要怎麼贏?', '綠seafood要出招了']
    inference_features = extractFeatures(new_text)
    infer = loaded_clf.predict(inference_features)
    print([LabeltoTeamsDict[i] for i in infer])