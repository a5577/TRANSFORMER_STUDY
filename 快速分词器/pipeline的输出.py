from transformers import pipeline

"""
实际上，NER pipeline 模型提供了多种组合 token 形成实体的策略，可以通过 aggregation_strategy 参数进行设置：

simple：默认策略，以实体对应所有 token 的平均分数作为得分，例如“Sylvain”的分数就是“S”、“##yl”、“##va”和“##in”四个 token 分数的平均；
first：将第一个 token 的分数作为实体的分数，例如“Sylvain”的分数就是 token “S”的分数；
max：将 token 中最大的分数作为整个实体的分数；
average：对应词语（注意不是 token）的平均分数作为整个实体的分数，例如“Hugging Face”就是“Hugging”（0.975）和 “Face”（0.98879）的平均值 0.9819，而 simple 策略得分为 0.9796。
"""

# grouped_entities=True 自动归类
model = pipeline("token-classification",aggregation_strategy = "max")

result = model("My name is Sylvain and I work at Hugging Face in Brooklyn.")
print(result)