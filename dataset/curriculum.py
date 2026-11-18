"""Verified arithmetic/reading/logic/format data with component-level holdouts."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import random

NAMES=['小东','小西','小南','小北','小春','小夏','小秋','小冬','小林','小海','小山','小河','小星','小月','小云','小雨','小雪','小风','小青','小白','小红','小蓝','小绿','小紫','小晨','小夕','小松','小竹','小梅','小兰','小明','小亮','小光','小华','小芳','小英','小宁','小安','小平','小乐']
COLORS=['红色','蓝色','绿色','紫色','黄色','白色','黑色','橙色','粉色','灰色','棕色','青色']


def digest(text):return hashlib.sha256(text.encode('utf-8')).hexdigest()


def component_partition(key):
    slot=int(digest(key)[:12],16)%100
    return 'sft' if slot<75 else 'rl' if slot<90 else 'validation' if slot<95 else 'test'


def case(rng,kind):
    if kind=='arithmetic':
        operation=rng.choice(['+','-','*','/']);a,b=rng.randrange(100),rng.randrange(1,100)
        if operation=='+':answer=a+b
        elif operation=='-':answer=a-b
        elif operation=='*':answer=a*b
        else:answer=a;a=a*b
        symbol={'*':'×','/':'÷'}.get(operation,operation)
        first,second=sorted((a,b)) if operation in ('+','*') else (a,b)
        key=f'arithmetic:{first}:{operation}:{second}'
        variant=rng.randrange(4)
        prompt=[f'请计算 {a}{symbol}{b}，最后把答案放在\\boxed{{}}中。',
                f'{a}{symbol}{b} 等于多少？最后用\\boxed{{}}给出结果。',
                f'计算题：{a}{symbol}{b}。请先写出等式，再把最终答案放在\\boxed{{}}中。',
                f'求 {a}{symbol}{b} 的值，只输出\\boxed{{答案}}。'][variant]
        response=(f'{a}{symbol}{b}={answer}。\n' if variant==2 else '')+f'\\boxed{{{answer}}}'
        return dict(component=key,prompt=prompt,answer=str(answer),response=response,reward='math',domain='数学')
    if kind=='reading':
        names=rng.sample(NAMES,4);colors=rng.sample(COLORS,4);asked=rng.randrange(4)
        facts=list(zip(names,colors));rng.shuffle(facts)
        material='；'.join(f'{name}的杯子是{color}' for name,color in facts)+'。'
        prompt=f'材料：{material}\n问题：{names[asked]}的杯子是什么颜色？只回答颜色。'
        key='reading:'+json.dumps(sorted(zip(names,colors)),ensure_ascii=False)
        return dict(component=key,prompt=prompt,answer=colors[asked],response=colors[asked],reward='exact',domain='阅读')
    if kind=='logic':
        names=rng.sample(NAMES,4);lowest=rng.choice([False,True]);edges=list(zip(names,names[1:]));rng.shuffle(edges)
        relations='；'.join(f'{a}比{b}高' for a,b in edges)+'。'
        target=names[-1] if lowest else names[0];adjective='矮' if lowest else '高'
        prompt=f'{relations}\n四个人中谁最{adjective}？只回答名字。'
        return dict(component='logic:'+','.join(names),prompt=prompt,answer=target,response=target,reward='exact',domain='逻辑')
    name=rng.choice(NAMES);number=rng.randrange(10000);key=f'format:{name}:{number}'
    prompt=f'已知姓名为{name}，分数为{number}。请输出JSON，恰好包含“姓名”和“分数”两个字段，姓名是字符串，分数是整数。只输出JSON。'
    answer=json.dumps({'姓名':name,'分数':number},ensure_ascii=False)
    return dict(component=key,prompt=prompt,answer=answer,response=answer,reward='exact',domain='指令跟随')


def prepare(destination,counts,seed=20261117,protected=None):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from dataset.anneal import digest as prompt_digest
    root=Path(destination)
    if any(root.glob('*.parquet')):raise FileExistsError('Curriculum output already exists')
    root.mkdir(parents=True,exist_ok=True)
    protection=json.loads(Path(protected).read_text(encoding='utf-8')) if protected else {}
    protected_keys=set(protection.get('keys',[]));protected_components=set(protection.get('components',[]))
    rng=random.Random(seed);kinds=['arithmetic','reading','logic','format'];seen=set();components={split:set() for split in counts}
    stats={};ledger={};supervised=[]
    for split,count in counts.items():
        rows=[];category=collections.Counter();attempts=0
        while len(rows)<count:
            attempts+=1
            if attempts>count*100+10000:raise RuntimeError('Insufficient unique curriculum components')
            kind=kinds[len(rows)%len(kinds)];item=case(rng,kind)
            if item['component'] in protected_components:continue
            if component_partition(item['component'])!=split:continue
            key=prompt_digest(item['prompt'])
            if key in protected_keys or key in seen:continue
            seen.add(key);components[split].add(item['component']);category[kind]+=1
            messages=[dict(role='user',content=item['prompt'])]
            if split=='sft':
                messages.append(dict(role='assistant',content=item['response']))
                row=dict(messages=messages,source='verified_curriculum/'+kind,domain=item['domain'],license='generated',tokens=0,provided_split='train')
            else:
                row=dict(messages=messages,reward=item['reward'],answer=item['answer'],domain=item['domain'],category=kind,calculation=item['component'])
            rows.append(row);ledger[key]=dict(split=split,component=item['component'])
            if split=='validation':
                supervised.append(dict(messages=messages+[dict(role='assistant',content=item['response'])],
                      source='verified_curriculum/'+kind,domain=item['domain'],license='generated',tokens=0,provided_split='val'))
        if split=='sft':supervised.extend(rows)
        else:pq.write_table(pa.Table.from_pylist(rows),root/f'{split}.parquet',compression='zstd')
        stats[split]=dict(questions=len(rows),components=len(components[split]),categories=dict(category))
        print(json.dumps(dict(phase='curriculum_ready',split=split,**stats[split])),flush=True)
    for left in components:
        for right in components:
            if left!=right and components[left]&components[right]:raise ValueError('Curriculum component leakage')
    pq.write_table(pa.Table.from_pylist(supervised),root/'sft.parquet',compression='zstd')
    (root/'ledger.json').write_text(json.dumps(ledger,ensure_ascii=False),encoding='utf-8')
    report=dict(seed=seed,stats=stats,protected_prompt_hashes=len(protected_keys),protected_components=len(protected_components),component_overlap=0,
                caveat='Procedural skill curriculum; not an independent source of encyclopedic facts.')
    (root/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return report


def prepare_quality_probes(destination):
    """Frozen basic-knowledge checks and open responses; never training targets."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    facts={
      'knowledge_science':[
        ('水分子整体是否带净电荷？只回答“带”或“不带”。','不带',[]),
        ('地球围绕哪颗恒星公转？只回答名称。','太阳',[]),
        ('太阳系中被称为“红色星球”的是哪颗行星？只回答名称。','火星',[]),
        ('声音能在真空中传播吗？只回答“能”或“不能”。','不能',[]),
        ('写出水的化学式，只输出化学式。','H2O',['H₂O']),
        ('写出氧气的化学式，只输出化学式。','O2',['O₂']),
        ('写出氢气的化学式，只输出化学式。','H2',['H₂']),
        ('植物光合作用主要吸收哪种气体？只回答名称。','二氧化碳',[]),
        ('植物光合作用释放哪种气体？只回答名称。','氧气',[]),
        ('电流的国际单位是什么？只回答单位中文名称。','安培',[]),
        ('电压的国际单位是什么？只回答单位中文名称。','伏特',[]),
        ('电阻的国际单位是什么？只回答单位中文名称。','欧姆',[]),
        ('力的国际单位是什么？只回答单位中文名称。','牛顿',[]),
        ('能量的国际单位是什么？只回答单位中文名称。','焦耳',[]),
        ('在空气中，光和声音哪一个传播更快？只回答“光”或“声音”。','光',[]),
        ('冰融化成水属于物理变化还是化学变化？只回答变化类别。','物理变化',[]),
        ('铁生锈属于物理变化还是化学变化？只回答变化类别。','化学变化',[])],
      'knowledge_biology':[
        ('人体推动血液循环的主要器官是什么？只回答名称。','心脏',[]),
        ('血液中主要运输氧气的是哪种细胞？只回答名称。','红细胞',[]),
        ('人体进行气体交换的主要器官是什么？只回答名称。','肺',['肺脏']),
        ('人体吸收营养物质的主要消化器官是什么？只回答名称。','小肠',[]),
        ('蝙蝠属于鸟类还是哺乳动物？只回答类别。','哺乳动物',['哺乳类']),
        ('鲸用肺还是鳃呼吸？只回答“肺”或“鳃”。','肺',[]),
        ('企鹅属于鸟类还是哺乳动物？只回答类别。','鸟类',[]),
        ('蛇属于脊椎动物还是无脊椎动物？只回答类别。','脊椎动物',[])],
      'knowledge_geography':[
        ('中国的首都是哪里？只回答城市名称。','北京',['北京市']),
        ('日本的首都是哪里？只回答城市名称。','东京',[]),
        ('法国的首都是哪里？只回答城市名称。','巴黎',[]),
        ('英国的首都是哪里？只回答城市名称。','伦敦',[]),
        ('地球上面积最大的海洋是什么？只回答名称。','太平洋',[])],
      'knowledge_units':[
        ('一年有几个月？只回答阿拉伯数字。','12',[]),
        ('一周有几天？只回答阿拉伯数字。','7',[]),
        ('一小时有多少分钟？只回答阿拉伯数字。','60',[]),
        ('一分钟有多少秒？只回答阿拉伯数字。','60',[]),
        ('一米等于多少厘米？只回答阿拉伯数字。','100',[]),
        ('一千克等于多少克？只回答阿拉伯数字。','1000',[]),
        ('一吨等于多少千克？只回答阿拉伯数字。','1000',[]),
        ('平面三角形的内角和是多少度？只回答阿拉伯数字。','180',[]),
        ('0摄氏度等于多少华氏度？只回答阿拉伯数字。','32',[])],
      'knowledge_literature':[
        ('鲁迅的原名是什么？只回答姓名。','周树人',[]),
        ('《红楼梦》前八十回的作者是谁？只回答姓名。','曹雪芹',[]),
        ('“床前明月光”出自李白的哪首诗？只回答诗题。','静夜思',['《静夜思》']),
        ('《春晓》的作者是谁？只回答姓名。','孟浩然',[]),
        ('“春眠不觉晓”的下一句是什么？只回答下一句。','处处闻啼鸟',[]),
        ('“床前明月光”的下一句是什么？只回答下一句。','疑是地上霜',[])],
      'translation':[
        ('把英文单词apple翻译为中文，只回答译词。','苹果',[]),
        ('把英文单词cat翻译为中文，只回答译词。','猫',[]),
        ('把英文单词water翻译为中文，只回答译词。','水',[]),
        ('把英文单词book翻译为中文，只回答译词。','书',['书本']),
        ('把英文单词red翻译为表示颜色的中文词，只回答译词。','红色',['红']),
        ('把英文单词blue翻译为表示颜色的中文词，只回答译词。','蓝色',['蓝'])],
      'basic_code':[
        ('Python表达式len([1, 2, 3])的结果是多少？只回答结果。','3',[]),
        ('Python中[]创建列表还是字典？只回答类别。','列表',[]),
        ('Python中{}创建列表还是字典？只回答类别。','字典',[]),
        ('Python语句print(1+1)会输出什么？只回答输出内容。','2',[]),
        ('字符串abc的第二个字符是什么？只回答该字符。','b',[])]}
    rows=[]
    def append(messages,category,answer=None,aliases=None,reference=None,reward='exact'):
        rows.append(dict(messages=messages,category=category,domain='通用',reward=reward,answer=answer,
                         answer_aliases=aliases or [],reference=reference))
    for category,items in facts.items():
        for question,answer,aliases in items:append([dict(role='user',content=question)],category,answer,aliases)
    for placement in [0,20,40]:
        lines=[f'第{i+1}条：工作人员整理图书，检查门窗，清点仓库，然后记录今天的工作进度。' for i in range(40)]
        lines.insert(placement,'重要记录：这次任务的唯一代号是青石蓝鹭。')
        append([dict(role='user',content='\n'.join(lines)+'\n这次任务的唯一代号是什么？只回答代号。')],f'long_context_{placement}','青石蓝鹭')
    append([dict(role='user',content='我的杯子是蓝色的。'),dict(role='assistant',content='你现在的杯子是蓝色的。'),
            dict(role='user',content='我刚把杯子换成了红色的。现在杯子是什么颜色？只回答颜色。')],'multiturn','红色')
    append([dict(role='user',content='请记住任务代号是青石。'),dict(role='assistant',content='任务代号是青石。'),
            dict(role='user',content='另一个任务叫白帆。我最开始让你记住的代号是什么？只回答代号。')],'multiturn','青石')
    append([dict(role='user',content='我有12个苹果。'),dict(role='assistant',content='你有12个苹果。'),
            dict(role='user',content='送走5个，又买了8个，现在有几个？只回答数字。')],'multiturn','15')
    for question,reference in [
        ('水分子有极性，是否意味着整个分子带负电？请解释。','整体电中性；氧氢之间的部分电荷分布不均导致极性。'),
        ('为什么不能从一次失败断定以后一定失败？','一次观察不足以证明所有未来情形；方法和条件可以变化。'),
        ('把“学而时习之，不亦说乎”翻译成白话。','学习后经常温习实践，也是一件令人快乐的事。'),
        ('写一段约80字的雨后校园描写。','切题、连贯、有具体感官细节；避免重复和无关内容。'),
        ('给一个刚搬到新城市的朋友提三条实际建议。','三条具体、可执行、与适应新城市有关的建议。'),
        ('写一个Python函数，返回两个数的和。','语法正确的函数，接收两个参数并return相加结果。'),
        ('你好，今天我有些紧张，能和我聊两句吗？','回应紧张情绪，自然接话；不编造个人经历。'),
        ('请用两句话介绍长江。','位于中国的长河等基本地理事实，恰好两句。'),
        ('续写：雨停了。小林推开教室的门，看见窗台上放着一本陌生的笔记本。','自然延续已给情节，人物与场景保持一致。'),
        ('请概括：小明提前出门，因为下雨公交晚点，最后仍准时到了学校。','小明提前出门，虽遇公交晚点仍准时到校。')]:
        append([dict(role='user',content=question)],'open_response',reference=reference,reward='judge')
    destination=Path(destination)
    if destination.exists():raise FileExistsError('Frozen quality probe set already exists')
    destination.parent.mkdir(parents=True,exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows),destination,compression='zstd')
    report=dict(questions=len(rows),categories=dict(collections.Counter(r['category'] for r in rows)),
                sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                protocol='Strict complete short answers with explicit aliases; open responses require semantic review. No candidate choices supplied.',
                caveat='Diagnostic authored basic-knowledge questions; overlap with common pretraining facts is expected. Repeated validation, not an untouched benchmark test.')
    destination.with_suffix('.manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True);parser.add_argument('--sft',type=int,default=60000)
    parser.add_argument('--rl',type=int,default=20000);parser.add_argument('--validation',type=int,default=1000)
    parser.add_argument('--test',type=int,default=1000);parser.add_argument('--protected-prompts');parser.add_argument('--seed',type=int,default=20261117)
    parser.add_argument('--quality-probes',action='store_true',help='Prepare frozen evaluation-only knowledge, dialogue and continuation probes')
    args=parser.parse_args(argv)
    if args.quality_probes:return prepare_quality_probes(args.out)
    return prepare(args.out,dict(sft=args.sft,rl=args.rl,validation=args.validation,test=args.test),args.seed,args.protected_prompts)


if __name__=='__main__':main()
