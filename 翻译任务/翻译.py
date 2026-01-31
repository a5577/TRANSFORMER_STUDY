import time
import numpy as np
from torch.utils.data import Dataset, random_split
import json
from sacrebleu.metrics import BLEU
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM
from torch import nn, optim
from transformers import AutoConfig
from transformers.models.marian import MarianPreTrainedModel, MarianModel


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

max_dataset_size = 220000
train_set_size = 200000
valid_set_size = 20000

class TRANS(Dataset):
    def __init__(self,data_file):
        self.data = self.load_data(data_file)
    def load_data(self,data_file):
        Data = {}
        with open(data_file,"rt",encoding="utf-8") as f:
            for idx,line in enumerate(f):
                if idx >= max_dataset_size:
                    break
                sample = json.loads(line.strip())
                Data[idx] = sample
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
data = TRANS('./translation2019zh/translation2019zh_train.json')
train_data,valid_data = random_split(data,[train_set_size,valid_set_size])
test_data = TRANS('./translation2019zh/translation2019zh_valid.json')
"""
print(f'train set size: {len(train_data)}')
print(f'valid set size: {len(valid_data)}')
print(f'test set size: {len(test_data)}')
print(next(iter(train_data)))
"""

# 数据预处理
checkpoint = "Helsinki-NLP/opus-mt-zh-en"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
"""
zh_sentence = train_data[0]["chinese"]
en_sentence = train_data[0]["english"]
inputs = tokenizer(zh_sentence)
targets = tokenizer(text_target=en_sentence)
# print(inputs)
"""
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)
max_length = 128
def collote_fn(batch_samples):
    batch_inputs,batch_targets = [], []
    for sample in batch_samples:
        batch_inputs.append(sample["chinese"])
        batch_targets.append(sample["english"])
    batch_data = tokenizer(
        batch_inputs,
        text_target=batch_targets,
        padding=True,
        max_length=max_length,
        truncation=True,
        return_tensors="pt"
    )
    # 移位
    batch_data['decoder_input_ids'] = model.prepare_decoder_input_ids_from_labels(batch_data['labels'])
    end_token_index = torch.where(batch_data['labels'] == tokenizer.eos_token_id)[1]
    for idx,end_idx in enumerate(end_token_index):
        batch_data['labels'][idx][end_idx+1:] = -100
    return batch_data
#定义模型
class MarianForMT(MarianPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = MarianModel(config)
        target_vocab_size = config.decoder_vocab_size
        self.register_buffer("final_logits_bias", torch.zeros((1, target_vocab_size)))
        self.lm_head = nn.Linear(config.d_model, target_vocab_size, bias=False)
        self.post_init()
    def forward(self,x):
        output = self.model(
            input_ids=x['input_ids'],
            attention_mask=x['attention_mask'],
            decoder_input_ids=x['decoder_input_ids'],
            decoder_attention_mask=x.get('decoder_attention_mask', None)
        )
        sequence_output = output.last_hidden_state
        lm_logits = self.lm_head(sequence_output)+self.final_logits_bias
        return lm_logits

# 模型训练
def train_model():
    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collote_fn)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)
    #criterion = nn.CrossEntropyLoss()
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
        print('epoch %3s loss: %.5f time %.2f' % (epoch + 1, total_loss /batch_num, time.time() - start))
    torch.save(model.state_dict(), './lyrics_model_%d.pth' % epochs)

# 测试函数
def test_model():
    bleu = BLEU()
    test_dataloader = DataLoader(valid_data, batch_size=32, collate_fn=collote_fn)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)
    #model.load_state_dict(torch.load('./lyrics_model_1.pth'))
    #model.load_state_dict(torch.load('./lyrics_model_1.pth'))
    preds,labels = [],[]
    for batch_data in tqdm(test_dataloader):
        batch_data = batch_data.to(device)
        with torch.no_grad():
            generated_tokens = model.generate(
                batch_data["input_ids"],
                attention_mask=batch_data["attention_mask"],
                max_length=max_length,
            ).cpu().numpy()
        label_tokens = batch_data["labels"].cpu().numpy()

        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        label_tokens = np.where(label_tokens != -100, label_tokens, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(label_tokens, skip_special_tokens=True)

        preds += [pred.strip() for pred in decoded_preds]
        labels += [[label.strip()] for label in decoded_labels]
    score = bleu.corpus_score(preds, labels)
    print("BLEU score:", score.score)

if __name__ == '__main__':
    """
    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collote_fn)
    batch = next(iter(train_dataloader))
    print(batch.keys())
    print('batch shape:', {k: v.shape for k, v in batch.items()})
    print(batch)
    
    config = AutoConfig.from_pretrained(checkpoint)
    model = MarianForMT.from_pretrained(checkpoint,config=config).to(device)
    print(model)
    """
    train_model()