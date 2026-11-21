"""Gradio front end for the text-to-SQL analyst (Hugging Face Space)."""
import logging
import os

import gradio as gr

import plain_pipeline as pp
from setup_database import setup_database

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("space")

MAX_CHARS = 300
pp.MODEL = os.getenv("MODEL", pp.MODEL)   # override from Space "Variables" without a code change
LLM = pp.hf_llm                           # swapped for a fake in tests

engine = setup_database()                 # builds job_insights.db, returns a READ-ONLY engine

EXAMPLES = [
    "How many job postings are in the dataset?",
    "What is the average salary for Data Scientists?",
    "Which industry has the highest average salary?",
    "How many remote-friendly jobs are there in each location?",
    "How many jobs have both high AI adoption and high automation risk?",
]


def ask(question: str) -> tuple[str, str]:
    """Returns (sql_used, report_or_message)."""
    question = (question or "").strip()
    if not question:
        return "", "Please type a question."
    if len(question) > MAX_CHARS:
        return "", f"Please keep questions under {MAX_CHARS} characters."
    if not os.getenv("HF_TOKEN"):
        return "", "This Space is missing its HF_TOKEN secret (Settings → Variables and secrets)."
    try:
        out = pp.answer(engine, question, llm=LLM)
    except RuntimeError as e:                       # no valid query after all retries
        log.warning("No valid query: %s", e)
        return "", "I couldn't turn that into a safe query on this dataset. Try rephrasing, or ask about the columns shown below."
    except Exception:
        log.exception("Model call failed")
        return "", "Something went wrong calling the model. Please try again in a moment."

    report = out["report"]
    if out["truncated"]:
        report += f"\n\n*Note: the query matched more than {pp.MAX_ROWS} rows, so this covers only the first {pp.MAX_ROWS}.*"
    return out["sql"], report


with gr.Blocks(title="Text2SQL Data Analyst") as demo:
    gr.Markdown(
        "# Text2SQL Data Analyst\n"
        "Ask a question about the **AI job market** dataset (500 postings). "
        "Queries are read-only and checked against the real schema before they run."
    )
    q = gr.Textbox(label="Your question", max_lines=3, placeholder="e.g. Which location has the most remote-friendly jobs?")
    btn = gr.Button("Ask", variant="primary")
    sql_out = gr.Code(label="SQL that was run", language="sql", interactive=False)
    report_out = gr.Markdown(label="Answer")
    gr.Examples(EXAMPLES, inputs=q)
    with gr.Accordion("Columns in the dataset", open=False):
        gr.Markdown("`Job_Title`, `Industry`, `Company_Size`, `Location`, `AI_Adoption_Level`, "
                    "`Automation_Risk`, `Required_Skills`, `Salary_USD`, `Remote_Friendly`, `Job_Growth_Projection`")
    btn.click(ask, q, [sql_out, report_out])
    q.submit(ask, q, [sql_out, report_out])

demo.queue(max_size=10, default_concurrency_limit=2)   # caps how much of your credits strangers can burn

if __name__ == "__main__":
    demo.launch()