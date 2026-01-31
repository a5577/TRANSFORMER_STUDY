"""
对于文本摘要任务，常用评估指标是 ROUGE 值 (short for Recall-Oriented Understudy for Gisting Evaluation)，
它可以度量两个词语序列之间的词语重合率。ROUGE 值的召回率表示参考摘要在多大程度上被生成摘要覆盖，
| 指标                | 含义                     |
| ----------------- | ---------------------- |
| **p (precision)** | 生成摘要中有多少是对的            |
| **r (recall)**    | 参考摘要中有多少被覆盖            |
| **f (F1)**        | precision 和 recall 的综合 |

"""
from rouge import Rouge

generated_summary = "I absolutely loved reading the Hunger Games"
reference_summary = "I loved reading the Hunger Games"

rouge = Rouge()

scores = rouge.get_scores(
    hyps=[generated_summary], refs=[reference_summary]
)[0]
print(scores)


