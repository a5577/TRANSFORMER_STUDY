from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "So have I!"
]

input = tokenizer(sequences)
print(input)

# 使用填充
"""
padding="longest"： 将序列填充到当前 batch 中最长序列的长度；
padding="max_length"：将所有序列填充到模型能够接受的最大长度，例如 BERT 模型就是 512。
"""

input1 = tokenizer(sequences, padding= "longest")
input2 = tokenizer(sequences, padding= "max_length")

print(input1)
print(input2)

# 使用截断
""""
截断操作通过 truncation 参数来控制，如果 truncation=True，那么大于模型最大接受长度的序列都会被截断，
例如对于 BERT 模型就会截断长度超过 512 的序列。此外，也可以通过 max_length 参数来控制截断长度：
"""

input3 = tokenizer(sequences,max_length=8,truncation=True)
print(input3)

# 返回值格式
"""
分词器还可以通过 return_tensors 参数指定返回的张量格式：设为 pt 则返回 PyTorch 张量；tf 则返回 TensorFlow 张量，np 则返回 NumPy 数组。
"""
input3 = tokenizer(sequences,padding=True,truncation=True,return_tensors="pt")
print(input3["input_ids"])
# 送入模型
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
output = model(**input3)
print(output.logits)

