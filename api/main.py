import os
from typing import Optional

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from approval.store import ApprovalStore
from memory.db import ExperimentDB
from reporting.report_generator import compute_kpis

# Reads from the SAME files the orchestrator process writes to (when both
# are run from the same working directory, which is the intended setup) -
# not a forked/duplicated data store. Overridable via env vars in case the
# API is run from a different directory than the orchestrator.
CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH", "./chroma_db")
APPROVAL_DB_PATH = os.environ.get("APPROVAL_DB_PATH", "approvals.db")
EVOLUTION_REPORT_PATH = os.environ.get("EVOLUTION_REPORT_PATH", "evolution_report.jsonl")

app = FastAPI(title="Autoresearch Dashboard")
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

db = ExperimentDB(db_path=CHROMA_DB_PATH)
store = ApprovalStore(db_path=APPROVAL_DB_PATH)


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    kpis = compute_kpis(db, store, evolution_report_path=EVOLUTION_REPORT_PATH)
    pending = store.list_pending()
    history = list(reversed(db.list_all_experiments(limit=50)))
    decided = [a for a in store.list_all(limit=50) if a["status"] != "pending"]
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {"kpis": kpis, "pending": pending, "history": history, "decided": decided},
    )


@app.post("/approvals/{request_id}/approve")
def approve(request_id: str, note: Optional[str] = Form(None)):
    store.decide(request_id, "approved", note=note)
    return RedirectResponse(url="/", status_code=303)


@app.post("/approvals/{request_id}/reject")
def reject(request_id: str, note: Optional[str] = Form(None)):
    store.decide(request_id, "rejected", note=note)
    return RedirectResponse(url="/", status_code=303)


@app.get("/api/pending")
def api_pending():
    return store.list_pending()


@app.get("/api/approvals")
def api_approvals(limit: int = 100):
    return store.list_all(limit=limit)


@app.get("/api/history")
def api_history(limit: int = 100):
    return list(reversed(db.list_all_experiments(limit=limit)))


@app.get("/api/report")
def api_report():
    return compute_kpis(db, store, evolution_report_path=EVOLUTION_REPORT_PATH)
