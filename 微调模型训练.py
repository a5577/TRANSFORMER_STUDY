import time
from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset
import json
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch import nn, optim
from transformers import AutoConfig, AutoModel
from tqdm.auto import tqdm
from transformers import BertPreTrainedModel, BertModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')
# 加载分词器
checkpoint = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# 构建数据集
class AFQMC(Dataset):
    def __init__(self,data_file):
        self.data = self.load_data(data_file)

    def load_data(self,data_file):
        Data = {}
        with open(data_file, 'r', encoding='utf-8') as f :
            for idx ,line in enumerate(f):
                sample = json.loads(line.strip())
                Data[idx] = sample
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
train_data = AFQMC('./afqmc_public/train.json')
valid_data = AFQMC('./afqmc_public/dev.json')

# print(train_data[0])

# 编码函数实现
def collote_fn(batch_samples):
    batch_sentence_1 ,batch_sentence_2 = [],[]
    batch_label = []
    for sample in batch_samples:
        batch_sentence_1.append(sample['sentence1'])
        batch_sentence_2.append(sample['sentence2'])
        batch_label.append(int(sample['label']))
    x = tokenizer(
        batch_sentence_1,
        batch_sentence_2,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    y = torch.tensor(batch_label)
    return x,y
"""
train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=collote_fn)
print(train_dataloader)
batch_X, batch_y = next(iter(train_dataloader))
print('batch_X shape:', {k: v.shape for k, v in batch_X.items()})
print('batch_y shape:', batch_y.shape)
print(batch_X)
print(batch_y)
"""

# 定义模型类
class BertForPairwiseCLS(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert_encoder = AutoModel.from_pretrained(checkpoint)
        self.dropout = nn.Dropout(0.1)
        self.output = nn.Linear(768,2)
    def forward(self,x):
        bert_output = self.bert_encoder(**x)
        cls_vectors = bert_output.last_hidden_state[:,0,:]
        cls_vectors = self.dropout(cls_vectors )
        output = self.output(cls_vectors)
        return output
"""
model = BertForPairwiseCLS().to(device)
print(model)
"""
learning_rate = 1e-5
batch_size = 4
epochs= 3
#训练函数
def train_model():
    model = BertForPairwiseCLS().to(device)
    train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=collote_fn)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=1e-5)
    epochs = 3
    start = time.time()
    num = 0
    total_loss = 0.0
    for epoch in range(epochs):
        for step,(x,y) in enumerate(tqdm(train_dataloader)):
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred,y)

            #梯度更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num +=1
            total_loss += loss.item()
        print('epoch %3s loss: %.5f time %.2f' % (epoch + 1, total_loss / num, time.time() - start))
    torch.save(model.state_dict(), './lyrics_model_%d.pth' % epochs)

# train_model()

# 验证
def test_model():
    model = BertForPairwiseCLS().to(device)
    model.load_state_dict(torch.load('./lyrics_model_3.pth'))
    test_dataloader = DataLoader(valid_data, batch_size=4,  collate_fn=collote_fn)
    correct = 0
    size = len(test_dataloader.dataset)
    with torch.no_grad():
        for step,(x,y) in enumerate(tqdm(test_dataloader)):
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            correct += (y_pred.argmax(1) == y).type(torch.float).sum().item()
    acc = correct / size
    print(acc)
if __name__ == '__main__':
    test_model()

