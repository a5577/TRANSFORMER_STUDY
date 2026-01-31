from transformer_study.模型与分词器.处理多段文本 import tokenizer
"""
实际上，无论快速或慢速分词器都提供了按 chunk 切分文本的功能，只需要在截断文本时再添加额外的参数 return_overflowing_tokens=True。
"""

sentence = "This sentence is not too long but we are going to split it anyway."
inputs = tokenizer(
    sentence, truncation=True, return_overflowing_tokens=True, max_length=6, stride=2
)

for ids in inputs["input_ids"]:
    print(tokenizer.decode(ids))

sentence1 = "This sentence is not too long but we are going to split it anyway."
inputs1 = tokenizer(
    sentence1, truncation=True, return_overflowing_tokens=True, max_length=6, stride=2
)
print(inputs1.keys())
# print(inputs1)
# 输出为分割后的部分属于第几个句子
print(inputs1["overflow_to_sample_mapping"])

sentences2 = [
    "This sentence is not too long but we are going to split it anyway.",
    "This sentence is shorter but will still get split.",
]
inputs2 = tokenizer(
    sentences2, truncation=True, return_overflowing_tokens=True, max_length=6, stride=2
)
# print(inputs2)
print(inputs2["overflow_to_sample_mapping"])

