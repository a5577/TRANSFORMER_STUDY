from transformers import AutoTokenizer

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

num_added_toks = tokenizer.add_tokens(['[ENT_START]', '[ENT_END]'], special_tokens=True)

print("We have added", num_added_toks, "tokens")

sentence = 'Two [ENT_START] cars [ENT_END] collided in a [ENT_START] tunnel [ENT_END] this morning.'
# 添加后在进行分词
print(tokenizer.tokenize(sentence))


