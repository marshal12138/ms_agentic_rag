"""System prompt and query extraction for the tool-augmented research agent."""
import json
import re
from typing import Optional

PROMPT_TEMPLATE = """You are a tool-augmented research agent for wiki-based factoid question answering.

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


def build_prompt(question: str) -> str:
    return PROMPT_TEMPLATE.format(question=question)


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def extract_query(model_output: str) -> Optional[str]:
    """Extract the search query from the model's first <tool_call> block.

    Returns None when the model did not emit a tool call (per readme, such
    samples are skipped).
    """
    if not model_output:
        return None
    m = _TOOL_CALL_RE.search(model_output)
    if not m:
        return None
    block = m.group(1).strip()
    # Strip an accidental markdown fence if present.
    if block.startswith("```"):
        block = block.strip("`")
        block = block.split("\n", 1)[-1] if "\n" in block else block
    try:
        obj = json.loads(block)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or obj.get("name") != "search":
        return None
    args = obj.get("arguments")
    if not isinstance(args, dict):
        return None
    query = args.get("query")
    if not isinstance(query, str) or not query.strip():
        return None
    return query.strip()


if __name__ == "__main__":
    # Validate prompt building and query extraction without any services.
    prompt = build_prompt("What is the capital of England?")
    assert "Question: What is the capital of England?" in prompt
    assert "search(query: string)" in prompt
    print("[prompt] build_prompt OK, length =", len(prompt))

    valid_output = (
        "<reason>I need to confirm the capital.</reason>\n"
        '<tool_call>{"name": "search", "arguments": {"query": "capital of England"}}</tool_call>'
    )
    assert extract_query(valid_output) == "capital of England"
    print("[prompt] extract_query (tool call) ->", extract_query(valid_output))

    fenced_output = (
        "<reason>search</reason>\n"
        "<tool_call>```json\n"
        '{"name": "search", "arguments": {"query": "London facts"}}\n'
        "```</tool_call>"
    )
    assert extract_query(fenced_output) == "London facts"
    print("[prompt] extract_query (fenced) ->", extract_query(fenced_output))

    answer_only = "<reason>I know this.</reason>\n<answer>London</answer>"
    assert extract_query(answer_only) is None
    print("[prompt] extract_query (no tool call) ->", extract_query(answer_only))

    assert extract_query("") is None
    assert extract_query("plain text") is None
    print("[prompt] all assertions passed")
