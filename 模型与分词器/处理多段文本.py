import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

token = tokenizer(sequence,return_tensors="pt")

print(token)
print(token.keys())
print(token["input_ids"])

# 情感分析 模型输出
"""
[[-1.5607,  1.6123]] 代表负正情绪的分数 这句话代表正面情绪
"""
output = model(**token)
print(output.logits)

