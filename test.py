from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, BertModel
import torch
from tqdm import tqdm
from params import BATCH_SIZE, LEARN_RATE, EPOCH
# Mock data

train_texts = ['Let’s go lakers',
    '8:30開始打嗎',
    '湖人隊加油湖人隊加油湖人隊加油湖人隊加油',
    '我金加油',
    '你湖超穩',
    '三衛沒了',
    '阿肥加油',
    'DLO還上 續約有望',
    '終於肯打三峰線了喔',
    '奧斯卡對決話劇社',
    'dlo',
    '#救贖之戰']
train_labels = [1, 0, 1,2, 1, 0, 2, 2,1, 0, 0, 1]

test_texts = ['勇士加油！阿是金塊',
              '八村上了',
              '第一場擺這陣容就好了 唉..',
              '阿肥要橫掃了',
              '嗎？哦 是jv',
              '君子之戰',
              '試探完畢',
              '要不要先猜湖人罰球幾顆啊！我覺得30up',
              '放羌俠今天要秀啥下線真期待',
              '挖 湖人這陣容有高',
              '上啊啊肥']
test_labels = [2, 1, 0, 2, 0, 0, 1, 1, 2, 1, 2]

train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)

label2id = {0:'NONE', 1:'LAL', 2:'DEN'}
id2label = {'NONE':0, 'LAL':1, 'DEN':2}

# Dataset Class

class ReviewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx], dtype=torch.long) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)
    

# Building Model

class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = BertModel.from_pretrained("bert-base-chinese")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 4)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# rpc_init("172.30.17.166")
model = BERTClass()
model.to(device)

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params =  model.parameters(), lr=0.4)

def calculate_accu(big_idx, targets):
    n_correct = (big_idx==targets).sum().item()
    return n_correct

def train():
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    for _ in tqdm(range(10)):
        for _,data in enumerate(train_loader, 0):
            ids = data['input_ids'].to(device, dtype = torch.long)
            mask = data['attention_mask'].to(device, dtype = torch.long)
            targets = data['labels'].to(device, dtype = torch.long)
    
            outputs = model(ids, mask)
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_idx = torch.argmax(outputs.data, dim=1)
            n_correct += calculate_accu(big_idx, targets)
    
            nb_tr_steps += 1
            nb_tr_examples+=targets.size(0)
            
            # if _%5000==0:
            #     loss_step = tr_loss/nb_tr_steps
            #     accu_step = (n_correct*100)/nb_tr_examples 
            #     print(f"Training Loss per 5000 steps: {loss_step}")
            #     print(f"Training Accuracy per 5000 steps: {accu_step}")
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print(f'The Total Accuracy: {(n_correct*100)/nb_tr_examples}')
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Training Loss: {epoch_loss}")
    print(f"Training Accuracy: {epoch_accu}")

    return 

# Evaluate

def valid(model, testing_loader):
    model.eval()
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0    
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['input_ids'].to(device, dtype = torch.long)
            mask = data['attention_mask'].to(device, dtype = torch.long)
            targets = data['labels'].to(device, dtype = torch.long)
            outputs = model(ids, mask)
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_idx = torch.argmax(outputs.data, dim=1)
            n_correct += calculate_accu(big_idx, targets)

            nb_tr_steps += 1
            nb_tr_examples+=targets.size(0)
            
            # if _%5000==0:
            #     loss_step = tr_loss/nb_tr_steps
            #     accu_step = (n_correct*100)/nb_tr_examples
            #     print(f"Validation Loss per 100 steps: {loss_step}")
            #     print(f"Validation Accuracy per 100 steps: {accu_step}")
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Validation Loss Epoch: {epoch_loss}")
    print(f"Validation Accuracy Epoch: {epoch_accu}")
    
    return epoch_accu

# Driver

if __name__ == "__main__":
    # Tokenize data
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese", num_labels=3, id2label=id2label, label2id=label2id)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)
    
    # Transform to Dataset
    train_dataset = ReviewsDataset(train_encodings, train_labels)
    val_dataset = ReviewsDataset(val_encodings, val_labels)
    test_dataset = ReviewsDataset(test_encodings, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BERTClass()
    model.to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARN_RATE)
    
    train()
    torch.save(model, './results/trained_model.bin')
    
    test_model = torch.load('./results/trained_model.bin')
    acc = valid(test_model, test_loader)
    print("Accuracy on test data = %0.2f%%" % acc)
    
