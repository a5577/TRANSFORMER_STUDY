from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
example = "My name is Sylvain and I work at Hugging Face in Brooklyn."
encoding = tokenizer(example)
print(encoding.tokens())
"""
<追踪映射>
在上面的例子中，索引为 5 的 token 是“##yl”，它是词语“Sylvain”的一个部分，因此在映射回原文时不应该被单独看待。我们可以通过 word_ids() 函数来获取每一个 token 对应的词语索引：
可以看到，特殊 token [CLS] 和 [SEP] 被映射到 None，其他 token 都被映射到对应的来源词语。这可以为很多任务提供帮助，例如对于序列标注任务，就可以运用这个映射将词语的标签转换到 
token 的标签；对于遮蔽语言建模 (Masked Language Modeling, MLM)，就可以实现全词遮盖 (whole word masking)，将属于同一个词语的 token 全部遮盖掉。
"""
print(encoding.word_ids())

"""
词语/token ——》文本：通过 word_to_chars()、token_to_chars() 函数来实现，返回词语/token 在原文中的起始和结束偏移量。
"""

token_index = 5
print('the 5th token is:', encoding.tokens()[token_index])
# 获取在原文中的位置
start, end = encoding.token_to_chars(token_index)
print('corresponding text span is:', example[start:end])
# word_ids()返回对应的单词排列数
word_index = encoding.word_ids()[token_index] # 3
start, end = encoding.word_to_chars(word_index)
print('corresponding word span is:', example[start:end])
print("*"*35)
"""
词语 <-->token：实际上，词语和 token 之间可以直接通过索引直接映射，分别通过 token_to_word() 和 word_to_tokens() 来实现：
"""

token_index = 5
print('the 5th token is:', encoding.tokens()[token_index])
corresp_word_index = encoding.token_to_word(token_index)
print('corresponding word index is:', corresp_word_index)
start, end = encoding.word_to_chars(corresp_word_index)
print('the word is:', example[start:end])
start, end = encoding.word_to_tokens(corresp_word_index)
print('corresponding tokens are:', encoding.tokens()[start:end])