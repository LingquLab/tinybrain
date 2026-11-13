"""RL 的奖励。每种奖励是一个函数 reward(episode, task) -> float，task 是 RL 数据里的一行（dict）。

    math     对最终答案（task["answer"]）
    code     跑测试（task["tests"]，Python 代码，里面用 assert）
    poetry   诗词格律（task["form"]：五绝、七绝、五律、七律），可以和评审模型的分数加权
    judge    评审模型按评分细则打分（RLAIF）
    agent    工具调用任务：最后的回答对不对（task["answer"]）
    choice   选择题：选项字母对不对（choice.py）
    reading  阅读理解：和参考答案按字算 F1（match.py）
    cloze    诗词填空：要和标准答案完全一样（match.py）
    instruct 指令跟随：程序检查字数、格式、关键词等约束，满足的比例（instruct.py）
    logic    逻辑、算术、谜题：reasoning-gym 自带的判分（logic.py）

每行数据用 task["reward"] 指定用哪一种。
"""
