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

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
MODEL = "anthropic:claude-sonnet-5-5"  # any provider supported by init_chat_model
RECURSION_LIMIT = 40  # hard cap on agent steps (model + tool calls)
MAX_TOOL_OUTPUT_CHARS = 4000
OUTPUT_DIR = Path("outputs")

SYSTEM_PROMPT = """\
You are a careful data-wrangling agent. You modify CSV data held in pandas
DataFrames according to the user's instruction.

All DataFrames live in a dict called `dfs`, keyed by name (the CSV file stem).
Reference tables (loaded from the web, see below) live in a dict called `refs`;
treat them as read-only. In code you write, `pd`, `np`, `dfs` and `refs` are
already available. Do not import anything except from the standard library.
Never read or write files and never access the network from code.

Workflow:
1. Call `list_dataframes`, then `inspect_dataframe` on the relevant frames.
2. If the task needs outside knowledge (code lists, standards, mappings between
   standards, lookup values), do NOT rely on memory. Use `search_web` to find
   authoritative or well-maintained sources. You can then reference the information
   found in your transformations. Only for very small mappings may you write the
   values directly into your code.
3. Use `explore` to check assumptions BEFORE changing anything: value formats,
   whitespace/case, unique values, missing values, which values of the data have
   no match in the reference table, and whether the reference key is unique.
4. Make changes with `apply_transformation`, in small, focused steps, each with
   a short description. Inspect the result after each step to verify it. If a
   step was wrong, use `undo_last_transformation`.
5. When the instruction is fully satisfied, stop calling tools and reply with a
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


def load_csvs(paths: list[Path]) -> dict[str, pd.DataFrame]:
    return {re.sub(r"\W+", "_", p.stem): pd.read_csv(p) for p in paths}


def strip_fences(code: str) -> str:
    code = code.strip()
    code = re.sub(r"^```(?:python)?\s*", "", code)
    code = re.sub(r"\s*```$", "", code)
    return code.strip()


def truncate(text: str, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} chars]"


def exec_on(dfs: dict[str, pd.DataFrame], code: str) -> tuple[dict[str, pd.DataFrame], str]:
    """Run code against `dfs` (mutates the dict passed in). Returns (dfs, stdout)."""
    namespace = {"pd": pd, "np": np, "dfs": dfs}
    out = io.StringIO()
    with redirect_stdout(out):
        exec(code, namespace)  # noqa: S102  (see warning at top of file)
    result = namespace["dfs"]
    if not isinstance(result, dict) or not all(isinstance(v, pd.DataFrame) for v in result.values()):
        raise TypeError("`dfs` must remain a dict of pandas DataFrames.")
    return result, out.getvalue()


# --------------------------------------------------------------------------- #
# Session state shared by all tools
# --------------------------------------------------------------------------- #
@dataclass
class Step:
    description: str
    code: str


@dataclass
class Session:
    input_paths: list[Path]
    original: dict[str, pd.DataFrame]
    working: dict[str, pd.DataFrame] = field(default_factory=dict)
    steps: list[Step] = field(default_factory=list)
    # snapshots[i] = state *before* step i, enabling undo
    snapshots: list[dict[str, pd.DataFrame]] = field(default_factory=list)

    def __post_init__(self):
        self.working = copy.deepcopy(self.original)


def build_tools(session: Session) -> list:
    """Create tools bound to a session (closures keep the state out of the LLM's hands)."""

    @tool
    def search_web(query: str) -> str:
        """Searches the web for information.

        Args:
            query: search query to use for web search

        Returns:
            Formatted search results with titles, URLs, and snippets
        """
        try:
            from urllib.parse import quote_plus

            import requests

            # Using DuckDuckGo HTML search (no API key required)
            search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(search_url, headers=headers, timeout=10)
            response.raise_for_status()

            from bs4 import BeautifulSoup

            soup = BeautifulSoup(response.text, "html.parser")

            results = []
            for result in soup.find_all("a", class_="result__a")[:10]:
                title = result.get_text(strip=True)
                href = result.get("href", "")
                # Extract the actual URL from DuckDuckGo redirect
                snippet_elem = result.find_parent("a").find_next_sibling() if result.find_parent("a") else None
                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                results.append(f"Title: {title}\nURL: {href}\nSnippet: {snippet}\n")

            if not results:
                return "No search results found."

            return "\n".join(results)
        except Exception as e:
            return f"Search error: {str(e)}"

    @tool
    def list_dataframes() -> str:
        """List the currently loaded DataFrames with their shapes (reflects all committed changes)."""
        return "\n".join(f"dfs[{n!r}]: shape={df.shape}" for n, df in session.working.items())

    @tool
    def inspect_dataframe(name: str, n_rows: int = 5) -> str:
        """Show schema, null counts, sample rows and summary statistics for one DataFrame.

        Args:
            name: key in `dfs` (the CSV file stem).
            n_rows: number of sample rows to show (max 50).
        """
        if name not in session.working:
            return f"ERROR: no DataFrame named {name!r}. Available: {list(session.working)}"
        df = session.working[name]
        n_rows = max(1, min(int(n_rows), 50))
        info = io.StringIO()
        df.info(buf=info)
        try:
            desc = df.describe(include="all").T.to_string()
        except Exception:
            desc = "(describe() unavailable)"
        return truncate(
            f"shape: {df.shape}\n\n{info.getvalue()}\n"
            f"head({n_rows}):\n{df.head(n_rows).to_string()}\n\n"
            f"describe:\n{desc}"
        )

    @tool
    def explore(code: str) -> str:
        """Run READ-ONLY pandas code to investigate the data. Nothing is saved.

        The code runs on a throwaway copy of `dfs`, so any modifications are
        discarded. Use print() to see results. Use this to test assumptions
        before calling apply_transformation.

        Args:
            code: Python code using `dfs`, `pd`, `np`.
        """
        scratch = copy.deepcopy(session.working)
        try:
            _, out = exec_on(scratch, strip_fences(code))
        except Exception:
            return "ERROR:\n" + truncate(traceback.format_exc(limit=3))
        return truncate(out) if out.strip() else "(code ran fine but printed nothing; use print())"

    @tool
    def apply_transformation(description: str, code: str) -> str:
        """Run code that MODIFIES the data and commit it if it succeeds.

        On error the data is left untouched, so you can fix the code and retry.
        Committed code is logged and becomes part of the reproducible script,
        so it must be deterministic and depend only on `dfs`.

        Args:
            description: one short sentence describing this step.
            code: Python code that modifies the `dfs` dict in place or reassigns entries.
        """
        code = strip_fences(code)
        candidate = copy.deepcopy(session.working)
        try:
            candidate, out = exec_on(candidate, code)
        except Exception:
            return "ERROR (nothing was changed):\n" + truncate(traceback.format_exc(limit=3))

        session.snapshots.append(session.working)
        session.working = candidate
        session.steps.append(Step(description=description, code=code))

        shapes = ", ".join(f"{n}: {df.shape}" for n, df in session.working.items())
        msg = f"OK. Committed step {len(session.steps)}. Shapes now -> {shapes}"
        if out.strip():
            msg += f"\nstdout:\n{truncate(out, 1500)}"
        return msg

    @tool
    def undo_last_transformation() -> str:
        """Roll back the most recently committed transformation."""
        if not session.steps:
            return "Nothing to undo."
        removed = session.steps.pop()
        session.working = session.snapshots.pop()
        return f"Undid step: {removed.description!r}. {len(session.steps)} step(s) remain."

    return [search_web, list_dataframes, inspect_dataframe, explore, apply_transformation, undo_last_transformation]


# --------------------------------------------------------------------------- #
# Reproducible script + replay verification
# --------------------------------------------------------------------------- #
def build_repro_script(
    steps: list[Step],
    input_paths: list[Path],
    output_paths: dict[str, Path],
    instruction: str,
) -> str:
    inputs = ",\n    ".join(repr(str(p)) for p in input_paths)
    outputs = ",\n    ".join(f"{k!r}: {str(v)!r}" for k, v in output_paths.items())

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
        f"INPUT_PATHS = [\n    {inputs},\n]\n\n"
        f"OUTPUT_PATHS = {{\n    {outputs},\n}}\n\n"
        "# --- Load ---------------------------------------------------------------\n"
        'dfs = {re.sub(r"\\W+", "_", Path(p).stem): pd.read_csv(p) for p in INPUT_PATHS}\n\n'
        "# --- Transform (agent-generated) ----------------------------------------\n"
        f"{transform}\n"
        "# --- Save ---------------------------------------------------------------\n"
        "for name, df in dfs.items():\n"
        '    out = Path(OUTPUT_PATHS.get(name, f"{name}_modified.csv"))\n'
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
    output_csvs: dict[str, Path] = field(default_factory=dict)
    script_path: Path | None = None
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

    def run(self, csv_paths: list[str | Path], instruction: str) -> AgentResult:
        input_paths = [Path(p) for p in csv_paths]
        original = load_csvs(input_paths)
        session = Session(input_paths=input_paths, original=original)

        load_dotenv()

        llm = ChatOpenAI(
            temperature=0,
            model=os.environ.get("LLM_MODEL"),
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            openai_api_base=os.environ.get("OPENAI_BASE_URL"),
        )
        agent = create_agent(
            model=llm,
            tools=build_tools(session),
            system_prompt=SYSTEM_PROMPT,
        )

        names = ", ".join(f"{n!r} {df.shape}" for n, df in original.items())
        user_msg = f"Loaded DataFrames: {names}\n\nInstruction:\n{instruction}"

        try:
            state = agent.invoke(
                {"messages": [{"role": "user", "content": user_msg}]},
                config={"recursion_limit": self.recursion_limit},
            )
            summary = _message_text(state["messages"][-1])
        except Exception as e:  # includes GraphRecursionError
            return AgentResult(success=False, steps=session.steps, error=f"{type(e).__name__}: {e}")

        # --- Verify, then write outputs ---------------------------------- #
        replay_ok, replay_msg = replay(session)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_csvs = {}
        for name, df in session.working.items():
            out = self.output_dir / f"{name}_modified.csv"
            df.to_csv(out, index=False)
            output_csvs[name] = out

        script = build_repro_script(session.steps, input_paths, output_csvs, instruction)
        script_path = self.output_dir / "reproduce.py"
        script_path.write_text(script)

        return AgentResult(
            success=True,
            summary=summary,
            steps=session.steps,
            output_csvs=output_csvs,
            script_path=script_path,
            replay_ok=replay_ok,
            replay_message=replay_msg,
        )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    # parser = argparse.ArgumentParser(description="Modify CSVs with an agent.")
    # parser.add_argument("args", nargs="+", help="One or more CSV paths, then the instruction (last arg).")
    # parsed = parser.parse_args()

    # *csvs, instruction = parsed.args
    # if not csvs:
    #     parser.error("Provide at least one CSV path and an instruction.")

    csvs = ["datasets/covid_global.csv"]
    instruction = pd.read_csv("dataset_modifications/dataset_modifications.csv")

    instruction = instruction.loc[instruction["dataset"] == "covid-global"]
    result = CSVAgent().run(csvs, instruction)

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
