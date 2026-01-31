"""
rouge 库默认使用空格进行分词，因此无法处理中文、日文等语言，最简单的办法是按字进行切分，当然也可以使用分词器分词后再进行计算，否则会计算出不正确的 ROUGE 值：
"""
from rouge import Rouge

generated_summary = "我在苏州大学学习计算机，苏州大学很美丽。"
reference_summary = "我在环境优美的苏州大学学习计算机。"

rouge = Rouge()

TOKENIZE_CHINESE = lambda x: ' '.join(x)

scores = rouge.get_scores(
    hyps=[TOKENIZE_CHINESE(generated_summary)],
    refs=[TOKENIZE_CHINESE(reference_summary)]
)
print(scores)

wrong_scores = rouge.get_scores(
    hyps=[generated_summary],
    refs=[reference_summary]
)
print(wrong_scores)
