"""
DQ·INTEL — AI Agent
Wraps the Anthropic SDK for server-side chat with dataset context.
"""

import anthropic


def build_system_prompt(analysis: dict, recs: list) -> str:
    cols = analysis["columns"]
    top_issues = "\n".join(
        f"- {r['col']}: {r['issue']} [{r['impact']} impact]"
        for r in recs[:8]
    ) or "None detected"

    col_summary = ", ".join(
        f"{c['col']}({c['type']}, miss:{c['missing_pct']:.0f}%, outliers:{c['outliers']})"
        for c in cols[:15]
    )

    return f"""You are a concise, expert data quality analyst assistant embedded in DQ·INTEL.

You have already analysed the user's dataset. Here is the full summary:

DATASET: {analysis['file_name']}, {analysis['total_rows']:,} rows, {analysis['num_cols']} columns
DQS SCORE: {analysis['DQS']:.1f}/100
  Completeness:  {analysis['C']:.0f}
  Consistency:   {analysis['Co']:.0f}
  Validity:      {analysis['V']:.0f}
  Uniqueness:    {analysis['U']:.0f}
DUPLICATES: ~{analysis['dupes_estimated']:,} estimated
COLUMNS: {col_summary}
TOP ISSUES:
{top_issues}

Rules:
- Be direct and specific to THIS dataset, not generic advice.
- Keep responses under 150 words unless the user asks for a detailed breakdown.
- Use bullet points only when listing 3+ items.
- Never repeat the full summary back unless asked.
- If asked for Python code, provide clean pandas/numpy code snippets.
- Be honest about uncertainty — these are statistical heuristics, not ground truth."""


def stream_response(analysis: dict, recs: list, history: list, api_key: str):
    """
    Stream a response from Claude given chat history.
    history: list of {"role": "user"|"assistant", "content": str}
    Yields text chunks.
    """
    client = anthropic.Anthropic(api_key=api_key)
    system_prompt = build_system_prompt(analysis, recs)

    with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=600,
        system=system_prompt,
        messages=history[-12:],  # keep last 12 turns for context
    ) as stream:
        for text in stream.text_stream:
            yield text


def get_response(analysis: dict, recs: list, history: list, api_key: str) -> str:
    """
    Non-streaming version. Returns full response string.
    """
    return "".join(stream_response(analysis, recs, history, api_key))