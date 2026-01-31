from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

torch.manual_seed(0)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

"""
贪心搜索 (Greedy Search) 在每轮迭代时，即在时间步t，简单地选择概率最高的下一个词作为当前词，
贪心搜索最大的问题是由于每次都只选择当前概率最大的词，相当于是区部最优解，因此生成的序列往往并不是全局最优的。
"""

input_ids = tokenizer.encode('I enjoy walking with my cute dog', return_tensors='pt')
greedy_output = model.generate(input_ids,max_length = 50)
# print(greedy_output.shape)
# print(tokenizer.decode(greedy_output[0],skip_special_tokens=True))

"""
柱搜索 (Beam search) 在每个时间步都保留 num_beams 个最可能的词，最终选择整体概率最大的序列作为结果。
虽然柱搜索得到的序列更加流畅，但是输出中依然出现了重复片段。最简单的解决方法是引入 n-grams 惩罚，其在
每个时间步都手工将那些会产生重复 n-gram 片段的词的概率设为 0。例如，我们额外设置参数 no_repeat_ngram_size=2 就能使生成序列中不会出现重复的 2-gram 片段：
柱搜索会在每个时间步都保留当前概率最高的前 num_beams 个序列，因此我们还可以通过设置参数 num_return_sequences（<= num_beams）来返回概率靠前的多个序列：
柱搜索更适用于机器翻译或摘要等生成序列长度大致可预测的任务，而在对话生成、故事生成等开放式文本生成任务 (open-ended generation) 上表现不佳。虽然通过 n-gram 
或者其他惩罚项可以缓解“重复”问题，但是如何控制”不重复”和“重复”之间的平衡又非常困难。
"""
beam_outputs = model.generate(
    input_ids,
    max_length = 50,
    num_beams = 5,
    no_repeat_ngram_size = 2,
    num_return_sequences = 3,
    early_stopping = True
)
for j, seq in enumerate(beam_outputs):
    pass
    # print(f"Beam {j}: {tokenizer.decode(seq, skip_special_tokens=True)}")
"""
采样 (sampling) 最基本的形式就是从当前上下文的条件概率分布中随机地选择一个词作为下一个词，
看上去还不错，但是细读的话会发现不是很连贯，这也是采样生成文本的通病：模型经常会生成前后不连贯的片段。一种解决方式是通过降低 softmax 的温度 (temperature) 使得分布 
 更尖锐，即进一步增加高概率词出现的可能性和降低低概率词出现的可能性。
"""
sample_output = model.generate(
    input_ids,
    do_sample=True,
    max_length=50,
    top_k=0,
    temperature = 0.6
)
# print(sample_output)
# print(sample_output.shape)
# print(tokenizer.decode(sample_output[0],skip_special_tokens=True))

"""
类似于柱搜索，Top-K 采样在每个时间步都保留最可能的 K 个词，然后在这 K 个词上重新分配概率质量。
"""

top_k_output = model.generate(
    input_ids,
    do_sample=True,
    max_length=50,
    top_k=10
)
print(tokenizer.decode(top_k_output[0],skip_special_tokens=True))

""""
Top-p 对 Top-K 进行了改进，每次只从累积概率超过 p
 的最小的可能词集中进行选择，然后在这组词语中重新分配概率质量。这样，每个时间步的词语集合的大小就可以根据下一个词的条件概率分布动态增加和减少。
 
"""
top_p_output = model.generate(
    input_ids,
    do_sample=True,
    max_length=50,
    top_p=0.92,
    top_k=0
)
# print(tokenizer.decode(top_p_output[0],skip_special_tokens=True))
"""
最后，与贪心搜索类似，为了获得多个独立采样的结果，我们设置 num_return_sequences > 1，并且同时结合 Top-p 和 Top-K 采样：
"""
sample_outputs = model.generate(
    input_ids,
    do_sample=True,
    max_length=50,
    top_k=50,
    top_p=0.95,
    num_return_sequences=3
)
for j,seq in enumerate(sample_outputs):
    print(f"Beam {j}: {tokenizer.decode(seq, skip_special_tokens=True)}")