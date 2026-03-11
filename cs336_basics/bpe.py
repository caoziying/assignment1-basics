import os
import regex as re
from collections import Counter, defaultdict  # 1. 导入 defaultdict

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """给定输入语料库的路径，运行并训练一个 BPE 分词器，输出其生成的词汇表和合并规则。

    参数 (Args):
        input_path (str | os.PathLike): BPE 分词器训练数据的来源路径。
        vocab_size (int): 词汇表中的项目总数（包含特殊 Token 在内）。
        special_tokens (list[str]): 要添加到词汇表的特殊字符串 Token 列表。
            这些字符串将始终被保留为单一 Token，绝不进行拆分。
            如果这些特殊 Token 出现在 `input_path` 数据中，它们会和其他字符串受到同等对待。

    返回 (Returns):
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab (词汇表):
                训练后生成的词汇表映射，从 int (Token ID) 映射到 bytes (Token 字节串)。
            merges (合并规则):
                BPE 合并规则列表。每一项是由字节组成的元组 (<token1>, <token2>)，
                表示合并记录，排列顺序即为合并执行的顺序。
    """
    # 可合并次数
    sz = vocab_size - len(special_tokens) - 256
    
    base_pat = r"""'(?i:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    base_re = re.compile(base_pat)

    cnt = Counter()

    with open(input_path, "rb") as f:
        raw = f.read()

    text_content = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n").decode("utf-8")
    # text_content = raw.decode('utf-8')
    if special_tokens:
        escaped_tokens = sorted((re.escape(tok) for tok in special_tokens), key=len, reverse=True)
        special_re = re.compile("|".join(escaped_tokens))

        last = 0
        for m in special_re.finditer(text_content):
            # 先处理 special token 前面的普通文本
            if m.start() > last:
                chunk = text_content[last:m.start()]
                for x in base_re.finditer(chunk):
                    cnt[x.group().encode("utf-8")] += 1

            # special token 本身作为一个完整 pretoken
            cnt[m.group().encode("utf-8")] += 1
            last = m.end()

        # 处理最后一个 special token 后面的尾巴
        if last < len(text_content):
            chunk = text_content[last:]
            for x in base_re.finditer(chunk):
                cnt[x.group().encode("utf-8")] += 1
    else:
        for x in base_re.finditer(text_content):
            cnt[x.group().encode("utf-8")] += 1
    mp = {(i,): i for i in range(256)}
    # 3. 初始化词汇表 (vocab) 和特殊 Token 映射
    vocab = {i: bytes([i]) for i in range(256)}
    special_token_map = {}
    # 给特殊 Token 分配固定 ID，紧跟在 255 之后
    next_id = 256
    for st in special_tokens:
        st_bytes = st.encode('utf-8')
        special_token_map[st_bytes] = next_id
        vocab[next_id] = st_bytes
        next_id += 1
    
    # 4. 初始化待合并的 texts
    texts = []
    for c in cnt.keys(): # c 是 bytes
        if c in special_token_map:
            # 特殊 Token：直接转为单元素 ID 列表，它将自然免疫后续的相邻对合并
            texts.append([special_token_map[c]])
        else:
            # 普通词块：打散成基础字节 ID 列表
            texts.append(list(c))
    merges = []
    tc = Counter()
    for c, text in zip(cnt, texts):
        num = cnt[c]
        for x, y in zip(text, text[1:]):
            tc[(x, y)] += num
    while True:
        if not tc:
            break
        # 把 k (ID 对) 映射回 vocab 中的真实 bytes 进行字典序比较
        best_pair = max(tc, key=lambda k: (tc[k], vocab[k[0]], vocab[k[1]]))
        # best_pair = max(tc, key=lambda k: (tc[k], k))  (按照更新后的id进行排序  发生错误)
        if tc[best_pair] < 2 or sz == 0:
            break
        sz -= 1
        p0, p1 = best_pair
        vocab[next_id] = vocab[p0] + vocab[p1]
        merges.append((vocab[p0], vocab[p1]))
        mp[best_pair] = next_id
        tc[best_pair] = 0 # 置 0
        for c, i in zip(cnt, range(len(texts))):
            text = texts[i]
            new_text = []
            if p0 not in text:
                continue
            for j in range(len(text) - 1):
                if (text[j], text[j+1]) != best_pair:
                    tc[(text[j], text[j+1])] -= cnt[c]
            j = 0
            while j < len(text):
                if j + 1 < len(text) and (text[j], text[j+1]) == best_pair:
                    new_text.append(next_id)
                    j += 1
                else:
                    new_text.append(text[j])
                j += 1
            for j in range(len(new_text) - 1):
                tc[(new_text[j], new_text[j+1])] += cnt[c]
            texts[i] = new_text
        del tc[best_pair] # 删除，防止一堆为0的死键
        next_id += 1
    return vocab, merges