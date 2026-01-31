from transformers import BertModel
from transformers import AutoTokenizer

model = BertModel.from_pretrained("../model/bert-base-cased/")

# 加载分词器并保存
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# tokenizer.save_pretrained("./model/berts/")
# bert分词
sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)

# 编码
ids = tokenizer.convert_tokens_to_ids(tokens)

# encode 将上述两个步骤合并
idx = tokenizer.encode(sequence)

# 一般直接使用分词器
"""
token_type_ids:区分不同句子
attention_mask:是否有填充
"""
tokenizer_text = tokenizer("Using a Transformer network is simple")

# decode进行解码
decoded_string = tokenizer.decode([101, 7993, 170, 13809, 23763, 2443, 1110, 3014, 102])

print(tokens)
print(ids)
print(idx)
print(tokenizer_text)
print(decoded_string)