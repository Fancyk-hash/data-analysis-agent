"""
Data Analysis Agent - CSC 446 Mini-project D
Uses smolagents for tool orchestration + a local Qwen model.
"""

import os
import io
import traceback
import textwrap
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from smolagents import tool, CodeAgent, TransformersModel

# ──────────────────────────────────────────────────────────────────────────────
# Global state shared across tool calls within a session
# ──────────────────────────────────────────────────────────────────────────────
_df = None
_df_name = ""
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)


@tool
def load_dataset(filename: str) -> str:
    """Load a CSV file into memory and return a summary of its structure.

    Args:
        filename: Path to the CSV file to load (relative or absolute).
    """
    global _df, _df_name
    try:
        _df = pd.read_csv(filename)
        _df_name = os.path.basename(filename)
        buf = io.StringIO()
        _df.info(buf=buf)
        info_str = buf.getvalue()
        preview = _df.head(3).to_string(index=False)
        return (
            f"Loaded '{_df_name}' - {_df.shape[0]} rows x {_df.shape[1]} columns.\n\n"
            f"Column info:\n{info_str}\n\nFirst 3 rows:\n{preview}"
        )
    except FileNotFoundError:
        return f"File not found: '{filename}'. Please check the path."
    except Exception as e:
        return f"Error loading dataset: {e}"


@tool
def run_query(code: str) -> str:
    """Execute a pandas expression against the loaded dataframe and return results.

    The dataframe is available as the variable df inside the expression.

    Args:
        code: A Python/pandas code snippet. df refers to the loaded dataframe.
    """
    global _df
    if _df is None:
        return "No dataset loaded. Call load_dataset() first."
    try:
        local_vars = {"df": _df.copy(), "pd": pd, "np": np}
        exec_globals = {}
        try:
            result = eval(code, exec_globals, local_vars)
        except SyntaxError:
            exec(code, exec_globals, local_vars)
            result = local_vars.get("result", "Code executed (no result variable returned).")
        return str(result)
    except Exception as e:
        return f"Query error:\n{traceback.format_exc(limit=3)}"


@tool
def describe_column(column: str) -> str:
    """Return detailed statistics for a single column in the loaded dataset.

    Args:
        column: The exact name of the column to describe.
    """
    global _df
    if _df is None:
        return "No dataset loaded. Call load_dataset() first."
    if column not in _df.columns:
        close = [c for c in _df.columns if column.lower() in c.lower()]
        hint = f" Did you mean one of: {close}?" if close else ""
        return f"Column '{column}' not found.{hint}\nAvailable: {list(_df.columns)}"
    col = _df[column]
    missing = col.isna().sum()
    missing_pct = 100 * missing / len(col)
    lines = [
        f"Column: '{column}'  |  dtype: {col.dtype}",
        f"  Count      : {col.count()} (missing: {missing} = {missing_pct:.1f}%)",
        f"  Unique vals: {col.nunique()}",
    ]
    if pd.api.types.is_numeric_dtype(col):
        lines += [
            f"  Mean       : {col.mean():.4f}",
            f"  Median     : {col.median():.4f}",
            f"  Std dev    : {col.std():.4f}",
            f"  Min / Max  : {col.min()} / {col.max()}",
            f"  Skewness   : {col.skew():.4f}",
            f"  Kurtosis   : {col.kurtosis():.4f}",
            f"  Q1 / Q3    : {col.quantile(0.25):.4f} / {col.quantile(0.75):.4f}",
        ]
    else:
        top5 = col.value_counts().head(5).to_string()
        lines.append(f"  Top 5 values:\n{textwrap.indent(top5, '    ')}")
    return "\n".join(lines)


@tool
def plot(code: str) -> str:
    """Execute a matplotlib plotting snippet and save the figure to disk.

    The loaded dataframe is available as df. Always use fig, ax = plt.subplots() pattern.

    Args:
        code: Python code that creates a matplotlib figure using df and plt.
    """
    global _df
    if _df is None:
        return "No dataset loaded. Call load_dataset() first."
    try:
        plt.close("all")
        local_vars = {"df": _df.copy(), "pd": pd, "np": np, "plt": plt, "sns": sns}
        exec(code, {}, local_vars)
        fig = local_vars.get("fig", plt.gcf())
        path = os.path.join(PLOT_DIR, f"plot_{len(os.listdir(PLOT_DIR))}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close("all")
        return f"Plot saved to: {path}"
    except Exception:
        plt.close("all")
        return f"Plot error:\n{traceback.format_exc(limit=3)}"


@tool
def correlation_matrix(columns: str) -> str:
    """Compute pairwise Pearson correlations for selected numeric columns.

    Args:
        columns: Comma-separated column names, or 'all' to use every numeric column.
    """
    global _df
    if _df is None:
        return "No dataset loaded. Call load_dataset() first."
    try:
        if columns.strip().lower() == "all":
            sub = _df.select_dtypes(include="number")
        else:
            cols = [c.strip() for c in columns.split(",")]
            missing = [c for c in cols if c not in _df.columns]
            if missing:
                return f"Unknown columns: {missing}"
            sub = _df[cols].select_dtypes(include="number")
        if sub.shape[1] < 2:
            return "Need at least 2 numeric columns for a correlation matrix."
        corr = sub.corr()
        pairs = (
            corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            .stack()
            .sort_values(key=abs, ascending=False)
            .head(5)
        )
        pairs_str = "\n".join(f"  {a} <-> {b}: {v:.4f}" for (a, b), v in pairs.items())
        return f"Correlation matrix:\n{corr.to_string()}\n\nTop correlated pairs:\n{pairs_str}"
    except Exception as e:
        return f"Correlation error: {e}"


@tool
def detect_outliers(column: str) -> str:
    """Detect outliers in a numeric column using the IQR method.

    Args:
        column: Name of the numeric column to analyse.
    """
    global _df
    if _df is None:
        return "No dataset loaded. Call load_dataset() first."
    if column not in _df.columns:
        return f"Column '{column}' not found. Available: {list(_df.columns)}"
    col = _df[column].dropna()
    if not pd.api.types.is_numeric_dtype(col):
        return f"Column '{column}' is not numeric."
    q1, q3 = col.quantile(0.25), col.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = _df[(col < lower) | (col > upper)]
    return (
        f"Outlier analysis for '{column}':\n"
        f"  IQR        : {iqr:.4f}\n"
        f"  Lower fence: {lower:.4f}\n"
        f"  Upper fence: {upper:.4f}\n"
        f"  Outliers   : {len(outliers)} rows ({100*len(outliers)/len(_df):.1f}% of data)\n"
        f"  Indices    : {list(outliers.index[:20])}{'...' if len(outliers) > 20 else ''}"
    )


@tool
def suggest_analysis(focus: str) -> str:
    """Suggest relevant analyses based on the loaded dataset.

    Args:
        focus: Area of interest e.g. 'trends', 'outliers', 'relationships', or 'general'.
    """
    global _df, _df_name
    if _df is None:
        return "No dataset loaded. Call load_dataset() first."
    numeric_cols = list(_df.select_dtypes(include="number").columns)
    cat_cols = list(_df.select_dtypes(include=["object", "category"]).columns)
    suggestions = [f"Suggestions for '{_df_name}' (focus: {focus})\n"]
    if numeric_cols:
        suggestions.append(f"- Distributions: describe_column() on {numeric_cols[:3]}")
        if len(numeric_cols) >= 2:
            suggestions.append(f"- Correlations: correlation_matrix('all')")
        suggestions.append(f"- Outlier check: detect_outliers() on {numeric_cols[0]}")
        suggestions.append(f"- Histogram: plot() with df['{numeric_cols[0]}'].hist()")
    if cat_cols:
        suggestions.append(f"- Category counts: run_query(\"df['{cat_cols[0]}'].value_counts()\")")
        if numeric_cols:
            suggestions.append(f"- Group analysis: run_query(\"df.groupby('{cat_cols[0]}')['{numeric_cols[0]}'].mean()\")")
    return "\n".join(suggestions)


# ──────────────────────────────────────────────────────────────────────────────
# AGENT
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are DataBot, an expert data analysis assistant for CSC 446.
You help users explore CSV datasets through natural language by calling the available tools.

When you receive a question:
1. Think about which tool(s) are needed.
2. Call them in a logical order: load_dataset first if no dataset is loaded, then query/describe/plot.
3. Write correct pandas/matplotlib code when using run_query or plot.
4. Interpret results clearly in plain language after every tool call.
5. If a tool fails (e.g., wrong column name), explain what went wrong and try an alternative.
6. Handle follow-up questions that reference previous results.

Never make up data - only report what the tools return."""

def build_agent(model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    """Create and return a CodeAgent with all data-analysis tools."""
    model = TransformersModel(model_id=model_id)
    agent = CodeAgent(
        tools=[
            load_dataset,
            run_query,
            describe_column,
            plot,
            correlation_matrix,
            detect_outliers,
            suggest_analysis,
        ],
        model=model,
        max_steps=10,
        additional_authorized_imports=["pandas", "numpy", "matplotlib", "seaborn", "scipy"],
    )
    return agent


# ──────────────────────────────────────────────────────────────────────────────
# LOCAL TEST
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    agent = build_agent()
    test_queries = [
        "Load the dataset at sample_data/titanic.csv",
        "What's the average Fare by Pclass?",
        "Describe the Age column",
        "Are there any outliers in the Fare column?",
    ]
    print("=" * 60)
    print("DATA ANALYSIS AGENT - local test")
    print("=" * 60)
    for q in test_queries:
        print(f"\n>>> {q}")
        try:
            answer = agent.run(q)
            print(answer)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[Agent error] {e}")

            