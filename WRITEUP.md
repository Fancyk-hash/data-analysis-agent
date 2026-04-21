# Mini-project D: Data Analysis Agent - Write-up
**CSC 446 – Natural Language Processing**
**Due: April 21, 2026**

## 1. Which project did you choose and why?
I chose Option D: Data Analysis Agent. Data exploration is a 
natural fit for an agentic loop because every analysis follows 
the same pattern: load data, inspect it, query it, visualize it, 
and interpret results. This maps directly onto the 
perceive-decide-act-observe cycle we studied in class. I also 
chose it because data analysis is a practical skill - being able 
to ask questions about a dataset in plain English without writing 
code yourself is genuinely useful.

## 2. What design decisions did you make?

**Framework: smolagents**
I used smolagents CodeAgent instead of pure transformers because 
it handles the tool-calling loop automatically and lets me focus 
on designing the tools. The @tool decorator enforces type hints 
and docstrings automatically.

**7 tools (exceeds the minimum of 3):**
- load_dataset - loads a CSV and returns schema and preview
- run_query - executes pandas expressions against the dataframe
- describe_column - returns full statistics for any column
- plot - runs matplotlib code and saves the chart
- correlation_matrix - finds relationships between columns
- detect_outliers - IQR-based outlier detection
- suggest_analysis - recommends what to analyse next

**Model choice**
I started with Qwen/Qwen2.5-3B-Instruct as recommended in the 
slides, but my laptop ran out of RAM loading the 6GB model. I 
switched to Qwen/Qwen2.5-0.5B-Instruct which runs locally using 
TransformersModel. This keeps everything on the local machine 
with no external API dependency.

**Telegram interface**
Each chat session gets its own agent instance keyed by chat_id. 
The bot sends plots as actual images in Telegram. Slash commands 
(/load, /suggest, /help) give users a clear entry point.

## 3. Where did the agent fail and how did you handle it?

**Real problems encountered during development:**

| Problem | What happened | How I fixed it |
|---------|--------------|----------------|
| HfApiModel renamed | smolagents 1.24.0 renamed HfApiModel to InferenceClientModel | Updated the import |
| system_prompt removed | New smolagents removed system_prompt parameter from CodeAgent | Removed the parameter |
| pandas not authorized | CodeAgent blocks pandas imports by default | Added additional_authorized_imports parameter |
| Model too large | Qwen 3B crashed laptop RAM | Switched to 0.5B version |
| Wrong column name | Agent types wrong column name | Fuzzy match suggests correct name |
| No dataset loaded | Tools called before loading data | Every tool checks _df is None first |
| Bad pandas code | Agent writes invalid code | try/except returns error so agent retries |

## 4. What would you improve with more time?
1. Use a larger model (7B) for more accurate and faster responses
2. Let users send CSV files directly through Telegram as attachments
3. Add persistent memory so the bot remembers datasets between sessions
4. Support multiple datasets loaded at the same time
5. Add a natural language summary after every plot automatically
