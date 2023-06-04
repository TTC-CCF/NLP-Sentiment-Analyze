from loadData import readData, dataAugmentation, loadValidate
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer, AdamW
from sklearn.model_selection import train_test_split
from params import model_name,  NUM_LABELS, hidden_layer_size, BATCH_SIZE, LEARN_RATE, save_path, EPOCH
from tqdm import tqdm
from sklearn.metrics import classification_report




class BERTLSTM(nn.Module):
    def __init__(self):
        super(BERTLSTM, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        for param in self.bert.parameters():
            param.requires_grad = True
            
        self.loss = nn.BCELoss()
            
        self.lstm = nn.LSTM(input_size=768, hidden_size=256, num_layers=256)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(hidden_layer_size, NUM_LABELS)

    def forward(self, input_ids, attention_mask):
        out, _ = self.bert(input_ids, attention_mask)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        out = self.lstm(out)
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])  # 句子最后时刻的 hidden state
        return out

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

def preProcessing(reviews, labels):
    encoded = tokenizer(reviews, padding=True, truncation=True, return_tensors='pt')
    dataset = ReviewsDataset(encoded, labels)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return loader  

def train(model, train_loader):
    model.train()
    num_tr_step = 0
    total_loss = 0
    for iter in tqdm(range(EPOCH)):
        per_epoch_loss = 0
        for batch in train_loader:
            batch = {key: val.to(device) for key, val in batch.items()}
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = model.loss(outputs, labels)
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
      
if __name__ == "__main__":
    
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BERTLSTM()
    
    teams_reviews, sent_reviews, teams_labels, sent_labels = readData()
    v_reviews, v_teams_labels, v_sent_labels = loadValidate()

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
    
    validate_teams_loader = preProcessing(v_reviews, v_teams_labels)
    validate_sent_loader = preProcessing(v_reviews, v_sent_labels)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=LEARN_RATE)
    
    print('Training...')
    train(model, train_teams_loader)
    
    with torch.no_grad():
        trained_state_dict = torch.load(save_path)
        test_model = BERTLSTM()
        test_model.load_state_dict(trained_state_dict)
        test_model.to(device)
        
        print('Testing...')
        validate(test_model, test_teams_loader)
        
        print('Testing on validate data...')
        validate(test_model, test_teams_loader)