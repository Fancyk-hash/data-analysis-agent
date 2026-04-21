# DataBot - Data Analysis Agent
**CSC 446 Natural Language Processing - Mini-project D**

A conversational AI agent that answers questions about CSV 
datasets through Telegram. Built with smolagents and a local 
Qwen model.

## What it does
Send DataBot a message like:
- "What is the average age in this dataset?"
- "Are there outliers in the Fare column?"
- "Show me a histogram of Age"
- "What is the correlation between numeric columns?"

And it automatically writes the code, runs it, and sends 
you back the answer or chart.

## Tools
| Tool | Description |
|------|-------------|
| load_dataset | Loads a CSV file |
| run_query | Runs pandas expressions |
| describe_column | Full column statistics |
| plot | Creates and sends charts |
| correlation_matrix | Finds column relationships |
| detect_outliers | IQR-based outlier detection |
| suggest_analysis | Recommends what to analyse |

## How to run

### 1. Install dependencies
pip install -r requirements.txt

### 2. Set your tokens
setx HF_TOKEN "your_huggingface_token"
setx TELEGRAM_TOKEN "your_telegram_token"

### 3. Generate sample data
python sample_data/generate_sample_data.py

### 4. Test locally
python agent.py

### 5. Run the Telegram bot
python telegram_bot.py

## Project Structure
