"""Strict final-answer checks for generated reading, logic and JSON tasks."""
import json
import re
import unicodedata


def extract(text):
    text=text.strip()
    markers=list(re.finditer(r'(?:最终答案|答案)\s*(?:是|为|:|：)\s*',text))
    if markers:text=text[markers[-1].end():].strip()
    return text.rstrip('。.!！ \n\t')


def reward(episode,task):
    if not episode.finished:return 0.
    text=extract(episode.text);answer=str(task['answer']).strip()
    aliases=task.get('answer_aliases')
    if aliases:
        if isinstance(aliases,str):aliases=json.loads(aliases)
        normalize=lambda value:unicodedata.normalize('NFKC',str(value)).strip()
        return float(normalize(text) in {normalize(value) for value in [answer,*aliases]})
    if answer.startswith('{'):
        try:
            encode=lambda value:json.dumps(json.loads(value),ensure_ascii=False,sort_keys=True,separators=(',',':'))
            return float(encode(text)==encode(answer))
        except (ValueError,TypeError):return 0.
    return float(text==answer)
