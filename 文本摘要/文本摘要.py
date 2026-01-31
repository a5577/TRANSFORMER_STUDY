import time

from torch import optim
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

# mT5摘要模型
checkpoint =  "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)

max_input_length = 512
max_target_length = 64

max_dataset_size = 200000
class LCSTS(Dataset):
    def __init__(self,data_file):
        self.data = self.load_data(data_file)

    def load_data(self,data_file):
        Data = {}
        with open(data_file,'rt',encoding='utf-8') as f :
            for idx ,line in enumerate(f):
                if idx >=max_dataset_size:
                    break
                items = line.strip().split('!=!')
                assert len(items) == 2
                Data[idx] = {
                    'title': items[0],
                    'content': items[1]
                }
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

train_data = LCSTS('./lcsts_tsv/data1.tsv')
valid_data = LCSTS('./lcsts_tsv/data2.tsv')
test_data = LCSTS('./lcsts_tsv/data3.tsv')

# 数据预处理
"""
与翻译任务类似，摘要任务的输入和标签都是文本，这里我们同样使用分词器提供的
 as_target_tokenizer() 函数来并行地对输入和标签进行分词，并且同样将标签序列中填充的 pad 字符设置为 -100 以便在计算交叉熵损失时忽略它们，
 以及通过模型自带的 prepare_decoder_input_ids_from_labels 函数对标签进行移位操作以准备好 decoder input IDs：
"""
def collote_fn(batch_samples):
    batch_input, batch_target = [], []
    for sample in batch_samples:
        batch_input.append(sample['content'])
        batch_target.append(sample['title'])
    batch_data = tokenizer(
        batch_input,
        padding=True,
        max_length=max_input_length,
        truncation=True,
        return_tensors="pt"
    )
    # 切换处理模式
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch_target,
            padding=True,
            max_length=max_target_length,
            truncation=True,
            return_tensors="pt"
        )["input_ids"]
        # 移位操作
        batch_data['decoder_input_ids'] = model.prepare_decoder_input_ids_from_labels(labels)
        end_token_index = torch.where(labels == tokenizer.eos_token_id)[1]
        for idx,end_idx in enumerate(end_token_index):
            labels[idx][end_idx+1:] = -100
        batch_data['labels'] = labels
    return batch_data

# 封装数据
# train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=collote_fn)
# valid_dataloader = DataLoader(valid_data, batch_size=4, shuffle=False, collate_fn=collote_fn)

# 训练模型
def train_model():
    train_dataloader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=collote_fn)
    optimizer = optim.Adam(model.parameters(),lr=1e-5)
    epochs = 3
    # 定义参数
    total_loss = 0.0
    batch_num = 0  # 样本数
    start = time.time()
    for epoch in range(epochs):
        for batch,batch_data in enumerate(tqdm(train_dataloader)):
            batch_data = batch_data.to(device)
            output = model(**batch_data)
            loss = output.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_num+=1
            total_loss = loss.item()
        print('epoch %3s loss: %.5f time %.2f' % (epoch + 1, total_loss / batch_num, time.time() - start))
    torch.save(model.state_dict(), './lyrics_model_%d.pth' % epochs)

def test_model():
    pass


if __name__ == '__main__':
    """
    print(f'train set size: {len(train_data)}')
    print(f'valid set size: {len(valid_data)}')
    print(f'test set size: {len(test_data)}')
    print(next(iter(train_data)))
    
    inputs = tokenizer("我叫张三，在苏州大学学习计算机。")
    print(inputs)
    print(tokenizer.convert_ids_to_tokens(inputs.input_ids))
    
    
    batch = next(iter(train_dataloader))
    print(batch.keys())
    print('batch shape:', {k: v.shape for k, v in batch.items()})
    print(batch)
    """
    train_model()
