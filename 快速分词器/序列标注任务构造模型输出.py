from transformers import AutoTokenizer,AutoModelForSequenceClassification
import torch
checkpoint = "dbmdz/bert-large-cased-finetuned-conll03-english"

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

example = "My name is Sylvain and I work at Hugging Face in Brooklyn."

input = tokenizer(example, return_tensors="pt")
output = model(**input)

print(input["input_ids"].shape)
print(output)
print(output.logits.shape)
# 查看9个标签
"""
这里使用的是 IOB 标签格式，“B-XXX”表示某一种标签的开始，“I-XXX”表示某一种标签的中间，“O”表示非标签。
因此，该模型识别的实体类型共有 4 种：miscellaneous、person、organization 和 location。
"""
print(model.config.id2label)

probabilities = torch.nn.functional.softmax(output.logits, dim=-1)[0].tolist()
predictions = output.logits.argmax(dim=-1)[0].tolist()
print(predictions)