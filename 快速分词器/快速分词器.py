"""
实际上，Hugging Face 共提供了两种分分词器：

慢速分词器：Transformers 库自带，使用 Python 编写；
快速分词器：Tokenizers 库提供，使用 Rust 编写。
"""
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
example = "Hello world!"
encoding = tokenizer(example)
print(encoding)
print(type(encoding))
# 检查是不是快分词器
print('tokenizer.is_fast:', tokenizer.is_fast)
print('encoding.is_fast:', encoding.is_fast)