import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
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
from transformers import get_scheduler
from tqdm import tqdm
from loadData import readData, dataAugmention
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

logging.set_verbosity_error()

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
    
class FineTuneModel(torch.nn.Module):
    def __init__(self, num_labels):
        super(FineTuneModel, self).__init__()
        self.num_labels = num_labels
        self.model = AutoModel.from_pretrained(model_name)
        self.classifier = torch.nn.Linear(hidden_layer_size, num_labels)
        self.lossFunc = torch.nn.CrossEntropyLoss()
        self.optimizer = AdamW(self.model.parameters(), lr=LEARN_RATE)
    
        torch.nn.init.xavier_normal_(self.classifier.weight)
        
    def forward(self, input_ids, attention_mask):
        last_hidden_state = self.model(input_ids, attention_mask)
        last_hidden_state = torch.mean(last_hidden_state[0], 1)
        logit = self.classifier(last_hidden_state)
        logit = torch.nn.functional.relu(logit)
        return logit

def train(model):
    model.train()
    num_tr_step = 0
    total_loss = 0
    for _ in tqdm(range(EPOCH)):
        for batch in train_loader:
            batch = {key: val.to(device) for key, val in batch.items()}
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            model.optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = model.lossFunc(outputs, labels)
            total_loss += loss.item()
            
            loss.backward()
            model.optimizer.step()
            lr_scheduler.step()
            num_tr_step += 1
    
    print(f'Average loss: {total_loss/num_tr_step}')
    torch.save(model.state_dict(), save_path)

def validate(test_model):
    test_model.eval() 
    predict = []
    g_true = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = {key: val.to(device) for key, val in batch.items()}
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
                
            outputs = test_model(input_ids, attention_mask)
            pred = torch.nn.functional.softmax(outputs, dim=-1)
            big_idx = torch.argmax(pred, dim=-1)
            g_true += list(labels.detach().cpu().numpy())
            predict += list(big_idx.detach().cpu().numpy())
            
    print(classification_report(g_true, predict))
    
    

if __name__ == '__main__':
    model = FineTuneModel(NUM_LABELS)
    tokenizer = AutoTokenizer.from_pretrained(model_name, num_labels=NUM_LABELS)
    
    reviews, labels = readData()
    
    train_texts, test_texts, train_labels, test_labels = train_test_split(reviews, labels, stratify=labels)
    train_texts, train_labels = dataAugmention(train_texts, train_labels)
    
    train_encoded = tokenizer(train_texts, padding=True, truncation=True, return_tensors='pt')
    test_encoded = tokenizer(test_texts, padding=True, truncation=True, return_tensors='pt')
    
    train_dataset = ReviewsDataset(train_encoded, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    test_dataset = ReviewsDataset(test_encoded, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    num_training_steps = EPOCH * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=model.optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    
    train(model)
    
    with torch.no_grad():
        trained_state_dict = torch.load(save_path)
        test_model = FineTuneModel(trained_state_dict['classifier.weight'].size()[0])
        test_model.load_state_dict(trained_state_dict)
        test_model.to(device)
        
        validate(test_model)
        
        # use custom test case
        inference = ['太陽教練真的爛', '金塊這季進步很大，波特回歸補齊三分，勾登磨合了幾季這季也配合的不錯莫雷原本以為傷後會爛掉但看起來三分有以前的準度連季賽被別隊二陣血洗的爛替補也進入狀況了']
        encoded = tokenizer(inference, padding = True, truncation=True, return_tensors='pt')
        input_ids = torch.tensor(encoded['input_ids']).to(device)
        attention_mask = torch.tensor(encoded['attention_mask']).to(device)
        outputs = test_model(input_ids, attention_mask)
        prob = torch.nn.functional.softmax(outputs, dim=-1)
        pred = list(torch.argmax(prob, dim=1).detach().cpu().numpy())
        print([LabeltoTeamsDict[p] for p in pred])