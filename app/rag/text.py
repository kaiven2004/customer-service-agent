from __future__ import annotations

import re

_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")
_ASCII_WORD = re.compile(r"[a-z0-9]+")

# 中文 bigram 会把“怎么办”切成“怎么/么办”，这类高频虚词 bigram
# 会让无关问题也命中知识库，先显式去掉。生产环境可以直接换 jieba 分词。
STOP_BIGRAMS = {
    "怎么", "么办", "什么", "么样", "如何", "多少", "多久", "可以", "能否",
    "是否", "请问", "一下", "我的", "你的", "我们", "你们", "这个", "那个",
    "哪个", "为什", "么是", "就是", "还是", "但是", "如果", "因为", "所以",
}
STOP_TOKENS = {"的", "了", "是", "在", "我", "你", "他", "她", "它", "吗",
               "呢", "吧", "啊", "呀", "着", "过"}


def tokenize(text: str) -> list[str]:
    """中文按字符 bigram、英文按整词切分，供 BM25 与向量化共用。"""
    text = text.lower()
    tokens: list[str] = []
    for run in _CJK_RUN.findall(text):
        if len(run) == 1:
            if run not in STOP_TOKENS:
                tokens.append(run)
        else:
            for i in range(len(run) - 1):
                gram = run[i : i + 2]
                if gram not in STOP_BIGRAMS:
                    tokens.append(gram)
    tokens.extend(_ASCII_WORD.findall(text))
    return tokens
