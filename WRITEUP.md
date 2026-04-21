# Mini-project D: Data Analysis Agent — Write-up
**CSC 446 – Natural Language Processing**
**Due: April 21, 2026**

## 1. Which project did you choose and why?
I chose Option D: Data Analysis Agent. Data exploration is a 
natural fit for an agentic loop because every analysis follows 
the same pattern: load data, inspect it, query it, visualize it, 
and interpret results. This maps directly onto the 
perceive-decide-act-observe cycle we studied in class.

## 2. What design decisions did you make?

**Framework: smolagents**
I used smolagents CodeAgent instead of pure transformers because 
it handles the tool-calling loop automatically and lets me focus 
on designing the tools themselves.

**7 tools (exceeds the minimum of 3):**
- load_dataset — loads a CSV and returns schema and preview
- run_query — executes pandas expressions against the dataframe
- describe_column — returns full statistics for any column
- plot — runs matplotlib code and saves the chart
- correlation_matrix — finds relationships between columns
- detect_outliers — IQR-based outlier detection
- suggest_analysis — recommends what to analyse next

**Telegram interface**
Each chat session gets its own agent instance. The bot sends 
plots as actual images in Telegram, not just file paths. 
Slash commands (/load, /suggest, /help) give users a clear 
entry point.

**Model**
I used Qwen/Qwen2.5-0.5B-Instruct running locally via 
TransformersModel. This keeps everything on the local machine 
with no external API dependency, which matches the course 
philosophy of local models.

## 3. Where did the agent fail and how did you handle it?

| Failure | How I handled it |
|---------|-----------------|
| Wrong column name | Fuzzy match suggests correct name |
| No dataset loaded | Every tool checks and returns clear message |
| Bad pandas code | try/except returns error to agent to retry |
| Plot has no fig variable | Falls back to plt.gcf() |
| Model generates wrong path | Agent retries with corrected path |

The agent handles all these gracefully without crashing.

## 4. What would you improve with more time?
1. Let users send CSV files directly through Telegram
2. Use a larger model (7B) for more accurate tool selection
3. Add persistent memory so the bot remembers past sessions
4. Support multiple datasets loaded at the same time
5. Add a natural language summary after every plot
