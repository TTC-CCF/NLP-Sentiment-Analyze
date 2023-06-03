import torch
from transformers import (
    XLNetTokenizer,
    XLNetForSequenceClassification,
    AdamW, 
    logging)
from torch.utils.data import DataLoader
from params import (
    LEARN_RATE,
    EPOCH,
    NUM_LABELS,
    BATCH_SIZE,
    LabeltoTeamsDict,
    model_name,
    save_path,
    hidden_layer_size,
    )
from tqdm import tqdm
from loadData import readData, dataAugmentation
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

logging.set_verbosity_error()

model = XLNetForSequenceClassification.from_pretrained(model_name, num_labels=NUM_LABELS)
tokenizer = XLNetTokenizer.from_pretrained(model_name)
    


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
    for iter in tqdm(range(EPOCH)):
        per_epoch_loss = 0
        for batch in train_loader:
            batch = {key: val.to(device) for key, val in batch.items()}
            

            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            per_epoch_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            num_tr_step += 1
        print(f'Epoch {iter+1} loss: {per_epoch_loss}')
        total_loss += per_epoch_loss
    
    print(f'Average loss: {total_loss/num_tr_step}')
    torch.save(model.state_dict(), save_path)

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
            
    print(classification_report(g_true, predict))
    
def preProcessing(reviews, labels):
    encoded = tokenizer(reviews, padding=True, truncation=True, return_tensors='pt')
    dataset = ReviewsDataset(encoded, labels)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return loader

if __name__ == '__main__':
    
    teams_reviews, sent_reviews, teams_labels, sent_labels = readData()
    
    teams = train_test_split(teams_reviews, teams_labels, stratify=teams_labels)
    sent = train_test_split(sent_reviews, sent_labels, stratify=sent_labels)
    
    print('Augmenting train dataset...')
    # train data
    train_teams_texts, train_teams_labels = dataAugmentation(teams[0], teams[2])
    train_sent_texts, train_sent_labels = dataAugmentation(sent[0], sent[2])
    
    # test data
    test_teams_texts, test_teams_labels = teams[1], teams[3]
    test_sent_texts, test_sent_labels = sent[1], sent[3]
    
    # build data into data loader
    train_teams_loader = preProcessing(train_teams_texts, train_teams_labels)
    test_teams_loader = preProcessing(test_teams_texts, test_teams_labels)
    
    train_sent_loader = preProcessing(train_sent_texts, train_sent_labels)
    test_sent_loader = preProcessing(test_sent_texts, test_sent_labels)
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=LEARN_RATE)
    
    print('Training...')
    train(model, train_teams_loader)
     
    with torch.no_grad():
        trained_state_dict = torch.load(save_path)
        test_model = XLNetForSequenceClassification.from_pretrained(model_name, num_labels=NUM_LABELS)
        test_model.load_state_dict(trained_state_dict)
        test_model.to(device)
        
        print('Testing...')
        validate(test_model, test_teams_loader)
        
        # use custom test case
        inference = ['太陽', '賽爾提克', '金塊這季進步很大，波特回歸補齊三分，勾登磨合了幾季這季也配合的不錯莫雷原本以為傷後會爛掉但看起來三分有以前的準度連季賽被別隊二陣血洗的爛替補也進入狀況了']
        encoded = tokenizer(inference, padding = True, truncation=True, return_tensors='pt')
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        outputs = test_model(input_ids, attention_mask)
        prob = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred = list(torch.argmax(prob, dim=1).detach().cpu().numpy())
        print([LabeltoTeamsDict[p] for p in pred])