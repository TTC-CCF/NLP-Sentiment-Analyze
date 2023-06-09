import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW, 
    logging,
    )
from torch.utils.data import DataLoader
from params import (
    LEARN_RATE,
    EPOCH,
    NUM_LABELS,
    BATCH_SIZE,
    LabeltoTeamsDict,
    LabeltoSentDict,
    model_name,
    save_path,
    )
from tqdm import tqdm
from loadData import readData, dataAugmentation, loadValidate
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np


logging.set_verbosity_error()

model = AutoModelForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=NUM_LABELS)
tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')

class ReviewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

def train(model, train_loader):
    model.train()
    num_tr_step = 0
    total_loss = 0
    accu_process = []
    loss_process = []
    for iter in tqdm(range(EPOCH)):
        per_epoch_loss = 0
        per_epoch_size = 0
        accu = 0
        for batch in train_loader:
            batch = {key: val.to(device) for key, val in batch.items()}
            labels = batch['labels']
            optimizer.zero_grad()
            outputs = model(**batch)
            prob = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predict = torch.argmax(prob, dim=-1)
            loss = outputs.loss

            per_epoch_loss += loss.item()
            accu += (labels==predict).sum().item()
            per_epoch_size += len(labels)

            loss.backward()
            optimizer.step()
            
        accu = accu/per_epoch_size
        average_loss = per_epoch_loss/per_epoch_size
        print(f'Epoch {iter+1} Accuracy: {accu}')
        print(f'Epoch {iter+1} Average loss: {average_loss}')
        num_tr_step += per_epoch_size
        total_loss += per_epoch_loss
        loss_process.append(average_loss)
        accu_process.append(accu)
    
    print(f'Average loss: {total_loss/num_tr_step}')
    torch.save(model.state_dict(), save_path)
    np.save('./loss/aug_sent.npy', np.array(loss_process))
    np.save('./accu/aug_sent.npy', np.array(accu_process))

def validate(test_model, test_loader):
    test_model.eval() 
    predict = []
    g_true = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = {key: val.to(device) for key, val in batch.items()}
            
            outputs = test_model(**batch)
            pred = torch.nn.functional.softmax(outputs.logits, dim=-1)
            big_idx = torch.argmax(pred, dim=-1)
            g_true += list(batch['labels'].detach().cpu().numpy())
            predict += list(big_idx.detach().cpu().numpy())
            
    print(classification_report(g_true, predict, digits=4))
    
def preProcessing(reviews, labels):
    encoded = tokenizer(reviews, padding=True, truncation=True, return_tensors='pt')
    dataset = ReviewsDataset(encoded, labels)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return loader

if __name__ == '__main__':
    teams_reviews, sent_reviews, teams_labels, sent_labels = readData()
    v_reviews, v_teams_labels, v_sent_labels = loadValidate()
    
    teams = train_test_split(teams_reviews, teams_labels, stratify=teams_labels)
    sent = train_test_split(sent_reviews, sent_labels, stratify=sent_labels)
    
    print('Augmenting train dataset...')
    # train data
    train_teams_texts, train_teams_labels =teams[0], teams[2] #dataAugmentation(teams[0], teams[2])
    train_sent_texts, train_sent_labels = sent[0], sent[2] #dataAugmentation(sent[0], sent[2])
    
    # test data
    test_teams_texts, test_teams_labels = teams[1], teams[3]
    test_sent_texts, test_sent_labels = sent[1], sent[3]
    
    # build data into data loader
    train_teams_loader = preProcessing(train_teams_texts, train_teams_labels)
    test_teams_loader = preProcessing(test_teams_texts, test_teams_labels)
    
    train_sent_loader = preProcessing(train_sent_texts, train_sent_labels)
    test_sent_loader = preProcessing(test_sent_texts, test_sent_labels)
    
    validate_teams_loader = preProcessing(v_reviews, v_teams_labels)
    validate_sent_loader = preProcessing(v_reviews, v_sent_labels)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=LEARN_RATE, no_deprecation_warning=True)
    
    # print('Training...')
    # train(model, train_sent_loader)
    with torch.no_grad():
        trained_state_dict = torch.load(save_path)
        test_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=NUM_LABELS)
        test_model.load_state_dict(trained_state_dict)
        test_model.to(device)
        
        print('Testing...')
        validate(test_model, test_teams_loader)
        
        
        print('Testing on validate data...')
        validate(test_model, validate_teams_loader)
        
        # use custom test case
        inference = ['阿肥大三元根本信手拈來', '杰倫快離開吧…跟著鐵圖姆沒前途的', '姆斯最棒棒了']
        encoded = tokenizer(inference, padding = True, truncation=True, return_tensors='pt')
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        outputs = test_model(input_ids, attention_mask)
        prob = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred = list(torch.argmax(prob, dim=1).detach().cpu().numpy())
        print([LabeltoTeamsDict[p] for p in pred])
