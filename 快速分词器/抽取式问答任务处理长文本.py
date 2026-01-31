import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

model_checkpoint = "distilbert-base-cased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

"""
幸运的是，自动问答 pipeline 采取了一种将超过最大长度的上下文切分为文本块 (chunk) 的方式，
即使答案出现在长文末尾也依然能够成功地抽取出来：
"""

long_context = """
Transformers: State of the Art NLP

Transformers provides thousands of pretrained models to perform tasks on texts such as classification, information extraction, question answering, summarization, translation, text generation and more in over 100 languages. Its aim is to make cutting-edge NLP easier to use for everyone.

Transformers provides APIs to quickly download and use those pretrained models on a given text, fine-tune them on your own datasets and then share them with the community on our model hub. At the same time, each python module defining an architecture is fully standalone and can be modified to enable quick research experiments.

Why should I use transformers?

1. Easy-to-use state-of-the-art models:
  - High performance on NLU and NLG tasks.
  - Low barrier to entry for educators and practitioners.
  - Few user-facing abstractions with just three classes to learn.
  - A unified API for using all our pretrained models.
  - Lower compute costs, smaller carbon footprint:

2. Researchers can share trained models instead of always retraining.
  - Practitioners can reduce compute time and production costs.
  - Dozens of architectures with over 10,000 pretrained models, some in more than 100 languages.

3. Choose the right framework for every part of a model's lifetime:
  - Train state-of-the-art models in 3 lines of code.
  - Move a single model between TF2.0/PyTorch frameworks at will.
  - Seamlessly pick the right framework for training, evaluation and production.

4. Easily customize a model or an example to your needs:
  - We provide examples for each architecture to reproduce the results published by its original authors.
  - Model internals are exposed as consistently as possible.
  - Model files can be used independently of the library for quick experiments.

Transformers is backed by the three most popular deep learning libraries — Jax, PyTorch and TensorFlow — with a seamless integration between them. It's straightforward to train your models with one before loading them for inference with the other.
"""

question = "Which deep learning libraries back Transformers?"

# ✅ 改动点：使用 pipeline
qa_pipeline = pipeline(
    "question-answering",
    model=model,
    tokenizer=tokenizer
)

result = qa_pipeline(
    question=question,
    context=long_context
)

print(result)

# ================= 下面保持你原来的代码不变 =================

# 长文本分词设置
inputs = tokenizer(
    question,
    long_context,
    stride=128,          # 步长
    max_length=384,      # 分割最大长
    padding="longest",   # 在当前 batch 内，按最长的那条序列补 padding
    truncation="only_second",  # 只截断第二个序列（context），绝不截断 question
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
)

# 第一个代表忽略
_ = inputs.pop("overflow_to_sample_mapping")
offsets = inputs.pop("offset_mapping")

inputs = inputs.convert_to_tensors("pt")
#print(inputs["input_ids"].shape)

outputs = model(**inputs)

start_logits = outputs.start_logits
end_logits = outputs.end_logits
#print(start_logits.shape, end_logits.shape)

# mask
sequence_ids = inputs.sequence_ids()

mask = [i !=1 for i in sequence_ids ]
# print(mask)
mask[0] = False
# 转成张量然后升维
mask = torch.logical_or(torch.tensor(mask)[None], (inputs["attention_mask"] == 0))
# print(mask)
start_logits[mask] = -10000
end_logits[mask] = -10000
start_probabilities = torch.nn.functional.softmax(start_logits, dim=-1)
end_probabilities = torch.nn.functional.softmax(end_logits, dim=-1)
# print(start_probabilities.shape)
candidates = []
for start_probs,end_probs in zip(start_probabilities,end_probabilities):
    scores = start_probs[:,None] * end_probs[None,:]
    max_index = scores.argmax().item()
    start_index = max_index // scores.shape[0]
    end_index = max_index % scores.shape[0]

    score = scores[start_index,end_index].item()
    candidates.append((start_index, end_index, score))
for candidate,offset in zip(candidates,offsets):
    start_token, end_token, score = candidate
    start_char, _ = offset[start_token]
    _, end_char = offset[end_token]
    answer = long_context[start_char:end_char]
    result = {"answer": answer, "start": start_char, "end": end_char, "score": score}
    print(result)
