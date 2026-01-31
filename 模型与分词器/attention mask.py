"""
在进行 Padding 操作时，我们必须明确告知模型哪些 token 是我们填充的，它们不应该参与编码。这就需要使用到 Attention Mask 了，在前面的例子中相信你已经多次见过它了。
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence1_ids = [[200, 200, 200]]
sequence2_ids = [[200, 200]]
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]
# 标出填充的地方
batched_attention_masks = [
    [1,1,1],
    [1,1,0],
]

# 合并前后编码后结果相同

print(model(torch.tensor(sequence1_ids)))
print(model(torch.tensor(sequence2_ids)))

output = model(
    torch.tensor(batched_ids),
    attention_mask = torch.tensor(batched_attention_masks)
)

print(output.logits)