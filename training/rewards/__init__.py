"""RL 的奖励。每种奖励是一个函数 reward(episode, task) -> float，task 是 RL 数据里的一行（dict）。

    math     对最终答案（task["answer"]）
    code     跑测试（task["tests"]，Python 代码，里面用 assert）
    poetry   诗词格律（task["form"]：五绝、七绝、五律、七律），可以和评审模型的分数加权
    judge    评审模型按评分细则打分（RLAIF）
    agent    工具调用任务：最后的回答对不对（task["answer"]）

每行数据用 task["reward"] 指定用哪一种。
"""
