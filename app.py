import os
import datetime
from flask import Flask, render_template, request, redirect, url_for, send_file, flash, session
from dotenv import load_dotenv
from openai import OpenAI

# -------------------------
# Load environment variables
# -------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Please set OPENAI_API_KEY in your environment or .env file")

client = OpenAI(api_key=OPENAI_API_KEY)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "change_this_secret_in_prod")

# -------------------------
# Agent Implementations
# -------------------------
def knowledge_agent(query):
    system = "You are KnowledgeAgent: fetch and concisely summarize factual background relevant to the user's query."
    user_prompt = (
        f"Task: Provide concise factual background, definitions, and immediate context for:\n\n{query}\n\n"
        "Include up to 5 short bullet points or a short paragraph. Cite well-known sources (e.g., journals, official sites) "
        "when possible (name sources; do not provide links)."
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user_prompt}],
        max_tokens=700
    )
    return resp.choices[0].message.content.strip()


def analysis_agent(knowledge_text, query):
    system = "You are AnalysisAgent: perform deep analytical reasoning over provided knowledge and the user's question."
    user_prompt = (
        f"User Query: {query}\n\n"
        "Knowledge Summary:\n"
        f"{knowledge_text}\n\n"
        "Task: Produce a structured analysis: identify key themes, causal links, risks/opportunities, and at least 3 bullet point insights. "
        "If assumptions are needed, list them clearly. Use a numbered or bullet structure."
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user_prompt}],
        max_tokens=800
    )
    return resp.choices[0].message.content.strip()


def response_agent(analysis_text, query):
    system = "You are ResponseAgent: write clear, polished executive summaries and reports for non-technical readers."
    user_prompt = (
        f"User Query: {query}\n\n"
        "Analysis:\n"
        f"{analysis_text}\n\n"
        "Task: Produce a 3-part deliverable in HTML format (safe to embed on a webpage):\n"
        "1) Short Executive Summary (2-4 sentences)\n"
        "2) Detailed Findings (narrative, weave in bullets as needed)\n"
        "3) Recommended Next Steps (3 concise action items)\n\n"
        "Return valid HTML fragments (<h3>, <p>, <ul>, <li>) only."
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user_prompt}],
        max_tokens=900
    )
    return resp.choices[0].message.content.strip()


# -------------------------
# Orchestrator
# -------------------------
def orchestrator(user_query):
    timestamp = datetime.datetime.utcnow().isoformat()
    results = {"query": user_query, "timestamp": timestamp}

    # Step 1: Knowledge
    knowledge = knowledge_agent(user_query)
    results["knowledge"] = knowledge

    # Step 2: Analysis
    analysis = analysis_agent(knowledge, user_query)
    results["analysis"] = analysis

    # Step 3: Response
    final_html = response_agent(analysis, user_query)
    results["final_html"] = final_html

    return results


# -------------------------
# Routes
# -------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/run", methods=["POST"])
def run():
    query = request.form.get("query", "").strip()
    if not query:
        flash("Please enter a query to run the agents.", "warning")
        return redirect(url_for("index"))

    try:
        results = orchestrator(query)
        # store in session for later report download
        session["agents_data"] = results
    except Exception as e:
        app.logger.exception("Error running orchestrator")
        flash(f"Error running agents: {e}", "danger")
        return redirect(url_for("index"))

    return render_template("result.html", results=results)


@app.route("/download_report", methods=["POST"])
def download_report():
    results = session.get("agents_data")
    if not results:
        flash("No report available. Please run the agents first.", "warning")
        return redirect(url_for("index"))

    os.makedirs("reports", exist_ok=True)
    file_path = os.path.join("reports", "multi_agent_report.html")

    report_html = render_template("result.html", results=results)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(report_html)

    return send_file(file_path, as_attachment=True)


# -------------------------
# Run App
# -------------------------
if __name__ == "__main__":
    app.run(debug=True, port=8501)
