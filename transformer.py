import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader
from params import LEARN_RATE, EPOCH, NUM_LABELS, BATCH_SIZE
from transformers import get_scheduler
from tqdm import tqdm
from loadData import readData
from sklearn.model_selection import train_test_split


model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=NUM_LABELS)
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', num_labels=NUM_LABELS)

reviews, labels = readData()

train_texts, test_texts, train_labels, test_labels = train_test_split(reviews, labels, test_size=0.3)

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

train_encoded_inputs = tokenizer(train_texts, padding=True, truncation=True, return_tensors='pt')

test_encoded_inputs = tokenizer(test_texts, padding=True, truncation=True, return_tensors='pt')

train_dataset = ReviewsDataset(train_encoded_inputs, train_labels)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = ReviewsDataset(test_encoded_inputs, test_labels)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

optimizer = AdamW(model.parameters(), lr=LEARN_RATE, no_deprecation_warning=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


num_training_steps = EPOCH * len(train_loader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

def train():
    model.train()
    num_tr_step = 0
    total_loss = 0
    for _ in tqdm(range(EPOCH)):
        for batch in train_loader:
            batch = {key: val.to(device) for key, val in batch.items()}
    
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            num_tr_step += 1
    
    print(f'Average loss: {total_loss/num_tr_step}')
    torch.save(model, './results/trained_model.bin')

def validate(test_model):
    test_model.eval() 
    acc = 0
    n = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = {key: val.to(device) for key, val in batch.items()}
            labels = batch['labels']
            outputs = test_model(**batch)
            pred = torch.nn.functional.softmax(outputs.logits, dim=-1)
            big_idx = torch.argmax(pred, dim=1)
            
            acc += (big_idx==labels).sum().item()
            n += labels.size(0)
    print(f"The total accuracy: {acc*100/n}%")
    
    
if __name__ == '__main__':
    train()
    with torch.no_grad():
        test_model = torch.load('./results/trained_model.bin')
        validate(test_model)
        
        # use custom test case
        encoded = tokenizer(['阿肥加油!', '湖人一定輸阿'], padding = True, truncation=True)
        input_ids = torch.tensor(encoded['input_ids'])
        print(input_ids)
        mask = torch.tensor(encoded['attention_mask'])
        label = torch.tensor([[2, 2]])
        outputs = test_model(input_ids, attention_mask=mask, labels=label)
        pred = torch.nn.functional.softmax(outputs.logits, dim=-1)
        print(pred)
        print(torch.argmax(pred, dim=1))