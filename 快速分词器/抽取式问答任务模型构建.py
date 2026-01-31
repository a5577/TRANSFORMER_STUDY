from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

model_checkpoint = "distilbert-base-cased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

context = """
Transformers is backed by the three most popular deep learning libraries — Jax, PyTorch, and TensorFlow — with a seamless integration between them. It's straightforward to train your models with one before loading them for inference with the other.
"""
question = "Which deep learning libraries back Transformers?"

inputs = tokenizer(question, context, return_tensors="pt")
outputs = model(**inputs)

# print(inputs["input_ids"].shape)

start_logits = outputs.start_logits
end_logits = outputs.end_logits
# print(start_logits.shape, end_logits.shape)

"""
输出格式为：
question
[SEP]
context
[CLS]
"""
sequence_ids = inputs.sequence_ids()
print(sequence_ids)
mask = [i!=1 for i in sequence_ids]
# print(mask)
mask[0] = False
# 转成张量然后升维
mask = torch.tensor(mask)[None]
# print(mask)
start_logits[mask] = -10000
end_logits[mask] = -10000
start_probabilities = torch.nn.functional.softmax(start_logits, dim=-1)[0]
end_probabilities = torch.nn.functional.softmax(end_logits, dim=-1)[0]
# print(start_probabilities) 一维
"""
因此，我们首先通过构建矩阵计算所有的概率值，
然后将 $\text{index}{start} > \text{index}{end}$ 对应的值赋为 0 来遮蔽掉这些不应该出现的情况，这可以使用 Pytorch 自带的 torch.triu() 函数来完成，它会返回一个 2 维张量的上三角部分：
"""
# 升维
scores = start_probabilities[:, None] * end_probabilities[None, :]
scores = torch.triu(scores)
# print(scores)
# print(scores.shape)

max_index = scores.argmax().item()
start_index = max_index // scores.shape[1]
end_index = max_index % scores.shape[1]
# print(max_index)

inputs_with_offsets = tokenizer(
    question,
    context,
    return_offsets_mapping=True
)
# print(inputs_with_offsets)
offsets = inputs_with_offsets["offset_mapping"]
start,_ = offsets[start_index]
_,end = offsets[end_index]

result = {
    "answer": context[start:end],
    "start": start,
    "end": end,
    "score": float(scores[start_index, end_index]),
}
print(result)