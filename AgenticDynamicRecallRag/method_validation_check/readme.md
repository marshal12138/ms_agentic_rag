# 介绍

当前子项目是对策略的有效性验证项目

# 框架

验证包含如下流程：
1. 读取数据集，数据集中包含问题和标准答案
2. 对每个问题，调用模型使其产生检索query
3. 对检索的query进行召回
4. 使用不同的alpha对召回结果进行rrf排序
5. 比较在不同alpha的情况下，包含正确答案的文档所处的最终位置分别是top几

# 细节

## 模型调用细节
1. 调用模型时，通过url调用本地部署的模型
2. 调用模型使用的提示词如下：
```
"""You are a tool-augmented research agent for wiki-based factoid question answering.

Your task is to answer questions drawn from Wikipedia-style datasets.
The final answer is evaluated using exact match (EM) or token-level F1, so it must be short and precise.

You have ONE tool available:
- search(query: string) -> returns a list of Wikipedia passages

============================================================
CRITICAL OUTPUT FORMAT (MUST FOLLOW EXACTLY)
============================================================

For EVERY assistant turn, you MUST output EXACTLY TWO TAG BLOCKS in this order:

1) <reason> ... </reason>
2) EITHER:
   (A) <tool_call> ... </tool_call>
   OR
   (B) <answer> ... </answer>

No other text is allowed outside these tags.
Do NOT output <tool_response>. The environment will provide tool results separately.

Allowed patterns:
- <reason> ... </reason>
  <tool_call> ... </tool_call>

- <reason> ... </reason>
  <answer> ... </answer>

If you violate the format, your output is invalid.

============================================================
TOOL CALL JSON SCHEMA (STRICT)
============================================================

When calling the tool, the <tool_call> block MUST contain ONLY a valid JSON object:

<tool_call>
{{
  "name": "search",
  "arguments": {{
    "query": "<string>"
  }}
}}
</tool_call>

Rules:
- "name" MUST be exactly "search"
- "arguments" MUST be an object
- "query" MUST be a single string
- Do NOT add extra keys
- Do NOT wrap JSON in Markdown
- Do NOT include comments, trailing commas, or natural language

============================================================
GENERAL TOOL USAGE
============================================================

Use the search tool whenever additional evidence would help you determine the correct answer.
If you believe you already have sufficient information to answer correctly, answer directly.

You may use multiple search calls across turns.

============================================================
SEARCH GUIDELINES
============================================================

- Write search queries that are clear and specific to what you want to confirm or find.
- After receiving evidence, reassess whether you can answer; if not, search again with a refined query.

============================================================
REASONING CONTENT REQUIREMENTS
============================================================

Inside <reason>, you MUST:
- Briefly state what you are trying to do in this step
- Indicate whether you will search or answer now
- If searching: state what you want to find/confirm (high-level)
- If answering: state that you believe the information is sufficient

Keep <reason> concise and decision-oriented.
Do NOT include detailed chain-of-thought.
Do NOT include tool JSON inside <reason>.

============================================================
ANSWER REQUIREMENTS (STRICT: SHORT ANSWER)
============================================================

Inside <answer>, you MUST:
- Output ONLY the final answer string
- Do NOT include explanations, reasoning, or extra text
- Do NOT include citations, sources, or formatting
- Use a concise canonical form (Wikipedia-style when possible)

Examples of valid answers:
- Paris
- 1997
- George Washington
- The Lord of the Rings

If the expected answer type is a person/place/organization/title/date, output only that span.
If multiple surface forms are possible, output the most standard form.

============================================================
INTEGRITY
============================================================

- Do not fabricate facts.
- If you are uncertain, use search to verify.
- If evidence is conflicting, search again with a query that resolves the conflict.

============================================================
BEGIN
============================================================

Question: {question}
"""
```
3. 从模型的输出中提取出query的内容
4. 如果模型不输出工具调用，那就跳过这条数据
5. 调用的时候可以并行调用，后面召回也可以并行召回


## 召回细节
1. 进行召回时，也是调用本地的召回接口，请求体例子如下
```
curl http://localhost:9011/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "queries": ["England", "American"]
  }'
```
2. 请求体参数详解
    1. queries参数为数组，里面是需要进行召回的若干个参数，因此，可以固定参数进行一波召回
    2. 请求后返回的格式为{"result":list[list[list]]}， 最终返回一个三维数组，依次是：每个query的召回结果，而每个query召回的结果是一个长度为2的二维数组，里面的元素是使用不同召回方法召回的**文档id**。
3. 需要根据文档id所以到对应的文档，因此，在验证不同alpha对召回结果的影响时，优先进行排序，然后把所有需要召回的id文档进行召回，再判断文档是否是包含了答案的文档

## 验证详解

1. 验证的内容是：当前的召回是混合召回，包含bm25召回和向量召回，而计算最终结果的时候需要根据不同的召回器权重进行rrf排序（rrf的k参数默认取60），现在验证不同召回器权重对排序结果的影响，由于只有两个召回器，因此只取一个参数alpha，另外一个参数就是1-alpha
2. 判断文档是否包含正确答案的方法为：数据集中的数据已经包含了正确答案，因此，只要是包含了正确答案字符的文档，就是正确的文档
3. alpha从0取到1，步长为0.1
4. 在展示最终结果的时候，展示不同的alpha取值导致的文档在最终**top5**文档中的位置的数组，比如说正确答案的文档在第1，那就输出[1]，如果有多个正确文档，就把所有的位置都输出，比如说第2和第4，就输出[2,4],如果不包含了，就输出[-1]
5. 最终文件的输出结果就是:{"原来的数据集相关的键值对", "alpha_1":[所在位置], "alpha_2":[所在位置]....}

## 其他细节
1. 收集文档的时候，参考下面的函数
```
import datasets
def load_corpus(corpus_path: str):
    # 这个是读入文档文件的函数
    return datasets.load_dataset("json", data_files=corpus_path, split="train", num_proc=4)


def load_docs(corpus, doc_idxs):
    # 这个是从文档文件中读入对应文档下标的函数
    return [corpus[int(idx)] for idx in doc_idxs]
```
2. 数据集的位置在/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/co_search/raw_sets/nq/test.jsonl
3. 答案是字符串数组，只要是包含了其中某个字符串数组的文档，就算是包含了正确答案
4. 只验证100条数据（这个搞成参数吧），就是只要可以统计的alpha对最终结果的影响达到了100条，就不继续跑后面的内容了
5. 如果所有情况下正确答案都被挤掉了，那这条数据是无效数据

# 可视化报告
TODO:
1. 你想想怎么可视化才能表示出对于同一个query，不同的alpha对最终文档的位置有影响，影响包括如下两个方面：1. 是否会把正确答案挤出top5; 2. 如果没有挤出去，那对排序结果的影响
