"""
Agentic CSV modification (LangChain + pandas)

The agent works in a loop with tools instead of writing code in one shot:

    list_dataframes          -> see what's loaded
    inspect_dataframe        -> schema, nulls, sample rows, describe()
    explore                  -> run READ-ONLY pandas code (on a copy) to test hypotheses
    apply_transformation     -> run code that CHANGES the data; on success it is
                                committed and logged, on failure nothing changes
    undo_last_transformation -> roll back the latest committed step

Every committed transformation is logged. At the end we:
  1. write the modified CSV(s),
  2. assemble a standalone `reproduce.py` from the logged steps,
  3. REPLAY that script's steps on freshly loaded data and check the result is
     identical to what the agent produced (a real reproducibility check).

Install:
    pip install pandas numpy langchain langchain-anthropic pydantic
    export ANTHROPIC_API_KEY=...

Usage:
    python csv_agent_agentic.py data.csv "clean the date column, drop duplicate ids, add a revenue column"
    python csv_agent_agentic.py a.csv b.csv "merge b into a on id, keep only rows from 2024"

WARNING: Model-written code is run with exec() in your own process. Fine for
local experiments; for anything user-facing, run it in a real sandbox.
"""

from __future__ import annotations

import copy
import io
import logging
import os
import re
import textwrap
import traceback
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

fileLogger = logging.getLogger("localModifLogger")

# console_handler = logging.StreamHandler()
# console_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
# fileLogger.addHandler(console_handler)

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
RECURSION_LIMIT = 40  # hard cap on agent steps (model + tool calls)
MAX_TOOL_OUTPUT_CHARS = 4000
OUTPUT_DIR = Path("outputs")

SYSTEM_PROMPT = """\
You are a careful data-wrangling agent. You modify CSV data held in pandas
DataFrames according to the user's instructions. These are submitted in the user prompt as a table,
where each row denotes the instruction for one column of the original data.

The CSV data lives in a DataFrame called `df`.
In code you write, `pd`, `np`, `df` and `refs` are already available.
Do not import anything except from the standard library.
Never read or write files and never access the network from code.

Workflow:
1. Call `inspect_dataframe` on the relevant frame.
2. If the task needs outside knowledge (code lists, standards, mappings between
   standards, lookup values), do NOT rely on memory. Use `search_web` to find
   authoritative or well-maintained sources.
3. If `search_web` does not give you satisfying results, you can reuse the tool up to
   5 times with different queries. If there are then still no satisfying results, skip the instruction
   and report in your summary that you couldn't find the necessary data given the instruction.
4. Use `explore` to check assumptions BEFORE changing anything: value formats,
   whitespace/case, unique values, missing values, which values of the data have
   no match in the reference table, and whether the reference key is unique.
5. Make changes with `apply_transformation`, in small, focused steps, each with
   a short description. Inspect the result after each step to verify it. If a
   step was wrong, use `undo_last_transformation`.
6. When the instruction is fully satisfied, stop calling tools and reply with a
   short summary: what you changed, which source(s) you used (URLs), and any
   values that could not be mapped.

Rules:
- Only change what the instruction asks for. Preserve everything else.
- For lookups prefer `Series.map(dict_or_series)` so row order and row count are
  preserved. If you merge, first make sure the key is unique in the reference.
- Never invent mappings for unmatched values: leave them missing and report them.
- Transformation code must be deterministic and self-contained: it will be
  re-run later, in order, on freshly loaded data and reference tables. It may
  rely on earlier committed steps, but NOT on variables created in `explore`.
- Set a random seed if you use randomness. Write readable, commented code.
- If the instruction is impossible or ambiguous given the data and sources,
  say so in your final reply instead of guessing.
"""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def strip_fences(code: str) -> str:
    code = code.strip()
    code = re.sub(r"^```(?:python)?\s*", "", code)
    code = re.sub(r"\s*```$", "", code)
    return code.strip()


def truncate(text: str, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} chars]"


def exec_on(df: pd.DataFrame, code: str) -> tuple[dict[str, pd.DataFrame], str]:
    """Run code against `df` (mutates the dataframe passed in). Returns (ds, stdout)."""
    namespace = {"pd": pd, "np": np, "df": df}
    out = io.StringIO()
    with redirect_stdout(out):
        exec(code, namespace)  # noqa: S102  (see warning at top of file)
    result = namespace["df"]
    if not isinstance(result, pd.DataFrame):
        raise TypeError("`df` must remain a pandas DataFrames.")
    return result, out.getvalue()


def _extract_message_info(msg) -> dict:
    """Extract only the relevant info from a LangChain message."""
    info = {"role": "unknown", "content": "", "tool_calls": [], "tool_outputs": []}

    # Determine role
    msg_type = type(msg).__name__
    if msg_type in ("HumanMessage", "HumanMessageChunk"):
        info["role"] = "user"
    elif msg_type in ("AIMessage", "AIMessageChunk"):
        info["role"] = "assistant"
    elif msg_type in ("SystemMessage", "SystemMessageChunk"):
        info["role"] = "system"
    elif msg_type in ("ToolMessage", "ToolMessageChunk"):
        info["role"] = "tool"
        info["tool_outputs"].append(
            {
                "tool_call_id": getattr(msg, "tool_call_id", "unknown"),
                "content": str(msg.content)[:500],  # truncate
            }
        )

    # Extract content (handle both str and list formats)
    content = msg.content
    if isinstance(content, str):
        info["content"] = content
    elif isinstance(content, list):
        # Handle content blocks like [{"type": "text", "text": "..."}]
        info["content"] = "".join(block.get("text", "") if isinstance(block, dict) else str(block) for block in content)

    # Extract tool calls
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        for tc in msg.tool_calls:
            info["tool_calls"].append({"name": tc.get("name", "unknown"), "args": tc.get("args", {})})

    return info


def _format_messages_for_logging(messages) -> str:
    """Format a list of messages for clean logging."""
    lines = []
    for i, msg in enumerate(messages, 1):
        info = _extract_message_info(msg)
        prefix = f"[{i}] {info['role'].upper()}:"

        # Content
        if info["content"]:
            content_preview = info["content"][:300] + "..." if len(info["content"]) > 300 else info["content"]
            lines.append(f"{prefix} {content_preview}")
        else:
            lines.append(prefix)

        # Tool calls
        if info["tool_calls"]:
            for tc in info["tool_calls"]:
                args_str = str(tc["args"])[:100] + "..." if len(str(tc["args"])) > 100 else str(tc["args"])
                lines.append(f"    -> TOOL: {tc['name']}({args_str})")

        # Tool outputs
        if info["tool_outputs"]:
            for to in info["tool_outputs"]:
                output_preview = to["content"][:200] + "..." if len(to["content"]) > 200 else to["content"]
                lines.append(f"    <- RESULT: {output_preview}")

    return "\n".join(lines)


def _log_request_size(system_prompt: str, tools: list, user_msg: str, label: str = "initial") -> None:
    """Log the approximate size of the LLM request payload."""
    sp_size = len(system_prompt)

    # Estimate tool definition size (name + description + schema)
    tool_sizes = []
    for _tool in tools:
        tool_info = f"{_tool.name}: {_tool.description}"
        # Add argument schema size
        if hasattr(_tool, "args_schema") and _tool.args_schema:
            tool_info += f" args={_tool.args_schema.model_json_schema()}"
        tool_sizes.append(len(tool_info))

    tools_total = sum(tool_sizes)
    user_size = len(user_msg)
    total = sp_size + tools_total + user_size

    fileLogger.info(
        f"[{label} request size] "
        f"system_prompt={sp_size} chars, "
        f"tools={len(tools)} ({tools_total} chars), "
        f"user_msg={user_size} chars, "
        f"TOTAL ≈ {total} chars"
    )
    # Log individual tool sizes at debug level
    for i, (_tool, size) in enumerate(zip(tools, tool_sizes, strict=False)):
        fileLogger.debug(f"  tool[{i}] {_tool.name}: {size} chars")


# --------------------------------------------------------------------------- #
# Session state shared by all tools
# --------------------------------------------------------------------------- #
@dataclass
class Step:
    description: str
    code: str


@dataclass
class Session:
    input_path: Path
    original: pd.DataFrame
    working: pd.DataFrame = field(default_factory=pd.DataFrame)
    steps: list[Step] = field(default_factory=list)
    # snapshots[i] = state *before* step i, enabling undo
    snapshots: list[dict[str, pd.DataFrame]] = field(default_factory=list)

    def __post_init__(self):
        self.working = copy.deepcopy(self.original)


def build_tools(session: Session) -> list:
    """Create tools bound to a session (closures keep the state out of the LLM's hands)."""

    @tool
    def inspect_dataframe(n_rows: int = 5) -> str:
        """Show schema, null counts, sample rows and summary statistics for one DataFrame.

        Args:
            n_rows: number of sample rows to show (max 50).
        """
        fileLogger.info(f"tool call: inspect_dataframe(n_rows={n_rows})")
        df = session.working
        n_rows = max(1, min(int(n_rows), 50))
        info = io.StringIO()
        df.info(buf=info)
        try:
            desc = df.describe(include="all").T.to_string()
        except Exception:
            desc = "(describe() unavailable)"
        result = truncate(
            f"shape: {df.shape}\n\n{info.getvalue()}\n"
            f"head({n_rows}):\n{df.head(n_rows).to_string()}\n\n"
            f"describe:\n{desc}"
        )
        fileLogger.info(
            f"tool result: inspect_dataframe -> {result[:300]}..."
            if len(result) > 300
            else f"tool result: inspect_dataframe -> {result}"
        )
        return result

    @tool
    def explore(code: str) -> str:
        """Run READ-ONLY pandas code to investigate the data. Nothing is saved.

        The code runs on a throwaway copy of `df`, so any modifications are
        discarded. Use print() to see results. Use this to test assumptions
        before calling apply_transformation.

        Args:
            code: Python code using `df`, `pd`, `np`.
        """
        code_preview = code[:100].replace("\n", " ") + "..." if len(code) > 100 else code.replace("\n", " ")
        fileLogger.info(f"tool call: explore(code='{code_preview}')")
        scratch = copy.deepcopy(session.working)
        try:
            _, out = exec_on(scratch, strip_fences(code))
        except Exception:
            error_msg = "ERROR:\n" + truncate(traceback.format_exc(limit=3))
            fileLogger.info(f"tool result: explore -> {error_msg}")
            return error_msg
        result = truncate(out) if out.strip() else "(code ran fine but printed nothing; use print())"
        fileLogger.info(
            f"tool result: explore -> {result[:300]}..." if len(result) > 300 else f"tool result: explore -> {result}"
        )
        return result

    @tool
    def apply_transformation(description: str, code: str) -> str:
        """Run code that MODIFIES the data and commit it if it succeeds.

        On error the data is left untouched, so you can fix the code and retry.
        Committed code is logged and becomes part of the reproducible script,
        so it must be deterministic and depend only on `dfs`.

        Args:
            description: one short sentence describing this step.
            code: Python code that modifies the `df` dict in place or reassigns entries.
        """
        code_preview = code[:100].replace("\n", " ") + "..." if len(code) > 100 else code.replace("\n", " ")
        fileLogger.info(f"tool call: apply_transformation(description='{description}', code='{code_preview}')")
        code = strip_fences(code)
        candidate = copy.deepcopy(session.working)
        try:
            candidate, out = exec_on(candidate, code)
        except Exception:
            error_msg = "ERROR (nothing was changed):\n" + truncate(traceback.format_exc(limit=3))
            fileLogger.info(f"tool result: apply_transformation -> {error_msg}")
            return error_msg

        session.snapshots.append(session.working)
        session.working = candidate
        session.steps.append(Step(description=description, code=code))

        shape = session.working.shape
        msg = f"OK. Committed step {len(session.steps)}. Shapes now -> {shape}"
        if out.strip():
            msg += f"\nstdout:\n{truncate(out, 1500)}"
        fileLogger.info(
            f"tool result: apply_transformation -> {msg[:300]}..."
            if len(msg) > 300
            else f"tool result: apply_transformation -> {msg}"
        )
        return msg

    @tool
    def undo_last_transformation() -> str:
        """Roll back the most recently committed transformation."""
        fileLogger.info("tool call: undo_last_transformation()")
        if not session.steps:
            result = "Nothing to undo."
            fileLogger.info(f"tool result: undo_last_transformation -> {result}")
            return result
        removed = session.steps.pop()
        session.working = session.snapshots.pop()
        result = f"Undid step: {removed.description!r}. {len(session.steps)} step(s) remain."
        fileLogger.info(f"tool result: undo_last_transformation -> {result}")
        return result

    @tool
    def search_web(query: str, categories: str = "general") -> str:
        """Search the web using local SearXNG instance.

        Args:
            query: search query
            categories: search category (general, news, images, videos, it, science, files, music, maps, social media)

        Returns:
            Formatted search results with titles, URLs, and snippets
        """
        fileLogger.info(
            f"tool call: search_web_searxng(query='{query[:100]}...', categories='{categories}')"
            if len(query) > 100
            else f"tool call: search_web_searxng(query='{query}', categories='{categories}')"
        )
        try:
            import requests

            # Using local SearXNG instance
            search_url = "http://localhost:8080/search"
            params = {"q": query, "format": "json", "categories": categories}
            fileLogger.debug(f"search_web_searxng: URL={search_url}, params={params}")
            response = requests.get(search_url, params=params, timeout=10)
            fileLogger.debug(
                f"search_web_searxng: status_code={response.status_code}, response_length={len(response.text)}"
            )
            fileLogger.debug(f"Raw results: {response.text}")
            response.raise_for_status()

            data = response.json()
            results = []
            result_count = len(data.get("results", []))
            fileLogger.debug(f"search_web_searxng: found {result_count} results")

            for result in data.get("results", [])[:10]:
                title = result.get("title", "")
                url = result.get("url", "")
                content = result.get("content", "")
                results.append(f"Title: {title}\nURL: {url}\nSnippet: {content}\n")

            if not results:
                fileLogger.info("tool result: search_web_searxng -> No search results found.")
                return "No search results found."

            result_str = "\n".join(results)
            fileLogger.info(
                f"tool result: search_web_searxng -> {result_str[:300]}..."
                if len(result_str) > 300
                else f"tool result: search_web_searxng -> {result_str}"
            )
            return result_str
        except Exception as e:
            error_msg = f"Search error: {str(e)}"
            fileLogger.info(f"tool result: search_web_searxng -> ERROR: {error_msg}")
            return error_msg

    # return [search_web,
    return [search_web, inspect_dataframe, explore, apply_transformation, undo_last_transformation]


# --------------------------------------------------------------------------- #
# Reproducible script + replay verification
# --------------------------------------------------------------------------- #
def build_repro_script(
    steps: list[Step],
    input_path: Path,
    output_path: Path,
    instruction: str,
) -> str:
    input = input_path
    output = output_path

    step_blocks = []
    for i, s in enumerate(steps, 1):
        step_blocks.append(f"# --- Step {i}: {s.description}\n{s.code}\n")
    transform = "\n".join(step_blocks) if step_blocks else "# (no transformations were needed)\n"

    return (
        '"""\n'
        "Auto-generated by csv_agent_agentic.py\n\n"
        f"Instruction: {instruction}\n"
        '"""\n'
        "import re\n"
        "from pathlib import Path\n\n"
        "import numpy as np\n"
        "import pandas as pd\n\n"
        f"INPUT_PATH = [\n    {input},\n]\n\n"
        f"OUTPUT_PATH = {{\n    {output},\n}}\n\n"
        "# --- Load ---------------------------------------------------------------\n"
        'dfs = {re.sub(r"\\W+", "_", Path(p).stem): pd.read_csv(p) for p in INPUT_PATH}\n\n'
        "# --- Transform (agent-generated) ----------------------------------------\n"
        f"{transform}\n"
        "# --- Save ---------------------------------------------------------------\n"
        "for name, df in dfs.items():\n"
        '    out = f"{OUTPUT_PATH}_modified.csv"))\n'
        "    out.parent.mkdir(parents=True, exist_ok=True)\n"
        "    df.to_csv(out, index=False)\n"
        '    print(f"Wrote {out}  shape={df.shape}")\n'
    )


def replay(session: Session) -> tuple[bool, str]:
    """Re-run all committed steps on fresh data and compare with the agent's result."""
    try:
        fresh = copy.deepcopy(session.original)
        for s in session.steps:
            fresh, _ = exec_on(fresh, s.code)
    except Exception:
        return False, "Replay raised an error:\n" + traceback.format_exc(limit=3)

    if set(fresh) != set(session.working):
        return False, f"Replay produced different frames: {set(fresh)} vs {set(session.working)}"
    for name in fresh:
        if not fresh[name].equals(session.working[name]):
            return False, f"Replay result for {name!r} differs from the agent's result (non-deterministic code?)"
    return True, "Replay matches the agent's result."


# --------------------------------------------------------------------------- #
# Agent wrapper
# --------------------------------------------------------------------------- #
@dataclass
class AgentResult:
    success: bool
    summary: str = ""
    steps: list[Step] = field(default_factory=list)
    output_csv: pd.DataFrame = field(default_factory=dict)
    script: str | None = None
    replay_ok: bool | None = None
    replay_message: str = ""
    error: str | None = None


def _message_text(message) -> str:
    """Final AI message content may be a str or a list of content blocks."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)


class CSVAgent:
    def __init__(self, output_dir: Path = OUTPUT_DIR, recursion_limit: int = RECURSION_LIMIT):
        self.output_dir = output_dir
        self.recursion_limit = recursion_limit

    def run(self, name: str, csv_path: str, instruction: str) -> AgentResult:
        original = pd.read_csv(csv_path)
        session = Session(input_path=csv_path, original=original)

        load_dotenv()

        llm = ChatOpenAI(
            temperature=0,
            model=os.environ.get("LLM_MODEL"),
            openai_api_key=os.environ.get("WILLMA_API_KEY"),
            openai_api_base=os.environ.get("WILLMA_BASE_URL"),
            streaming=True,
        )
        agent = create_agent(
            model=llm,
            tools=build_tools(session),
            system_prompt=SYSTEM_PROMPT,
        )
        instr_text = instruction.drop(columns="dataset").to_csv(index=False)
        user_msg = (
            f"Columns/dtypes:\n{original.dtypes.to_string()}\n\n"
            f"Sample rows:\n{original.head(5).to_csv(index=False)}\n\n"
            f"Instructions (one row per column):\n{instr_text}"
        )

        try:
            _log_request_size(
                system_prompt=SYSTEM_PROMPT, tools=build_tools(session), user_msg=user_msg, label="initial"
            )
            fileLogger.info(f">>> LLM REQUEST <<<\n{user_msg}")
            state = agent.invoke(
                {"messages": [{"role": "user", "content": user_msg}]},
                config={"recursion_limit": self.recursion_limit},
            )
            fileLogger.info(f'>>> LLM RESPONSE <<<\n{_format_messages_for_logging(state["messages"])}')
            fileLogger.info("--------------------------------------------------------------")
            summary = _message_text(state["messages"][-1])

        # except GraphRecursionError as e:
        #     pass
        except Exception as e:  # includes GraphRecursionError
            return AgentResult(success=False, steps=session.steps, error=f"{type(e).__name__}: {e}")

        # --- Verify, then write outputs ---------------------------------- #
        replay_ok, replay_msg = replay(session)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_csv = session.working
        # out_df =
        output_path = self.output_dir / f"{name}_modified.csv"
        # output_csv.to_csv(output_path, index=False)

        script = build_repro_script(session.steps, csv_path, output_path, instruction)
        # script_path = self.output_dir / "reproduce.py"
        # script_path.write_text(script)

        return AgentResult(
            success=True,
            summary=summary,
            steps=session.steps,
            output_csv=output_csv,
            script=script,
            replay_ok=replay_ok,
            replay_message=replay_msg,
        )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    instruction = pd.read_csv("dataset_modifications/dataset_modifications_2.csv")

    input_path = "datasets/covid_indonesia.csv"
    dataset = pd.read_csv(input_path)
    instruction = instruction.loc[instruction["dataset"] == "covid-indonesia"]
    result = CSVAgent().run("covid-indonesia", dataset, instruction)

    if not result.success:
        print(f"Agent failed: {result.error}")
        print(f"({len(result.steps)} step(s) had been committed; nothing was written.)")
        raise SystemExit(1)

    print(f"Agent summary:\n{result.summary}\n")
    print(f"Committed {len(result.steps)} transformation step(s):")
    for i, s in enumerate(result.steps, 1):
        print(f"\n  Step {i}: {s.description}")
        print(textwrap.indent(s.code, "      "))
    print()
    for name, path in result.output_csvs.items():
        print(f"Modified CSV [{name}]: {path}")
    print(f"Reproducible script: {result.script_path}")
    print(f"Replay check: {'PASS' if result.replay_ok else 'FAIL'} - {result.replay_message}")


if __name__ == "__main__":
    # modifications = pd.read_csv("dataset_modifications/dataset_modifications.csv")
    # modifications_sel = modifications.loc[modifications['dataset'] == "covid-global"]
    # print(modifications_sel['transformation'].unique())
    main()
