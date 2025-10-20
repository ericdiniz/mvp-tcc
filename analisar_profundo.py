"""Análise Profunda (5-10 repositórios)

Coleta dados detalhados para cada repositório: histórico de commits, pull requests
merges, issues com label 'bug', conteúdo de workflows e mudanças em diretórios de teste.

Saídas: JSON com métricas por repositório, gráficos (PNG) e CSV.

Requer: um token GitHub em .env (GITHUB_TOKEN), bibliotecas requests, pandas, matplotlib.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import matplotlib.pyplot as plt

from dotenv import load_dotenv

# Carrega .env
load_dotenv()

LOGGER = logging.getLogger("analisar_profundo")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    LOGGER.critical("GITHUB_TOKEN não encontrado no .env")
    raise SystemExit(1)

HEADERS = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}
API_GRAPHQL = "https://api.github.com/graphql"
API_REST = "https://api.github.com"


@dataclass
class DeepMetrics:
    full_name: str
    stars: int
    forks: int
    pushed_at: str
    primary_language: Optional[str]
    workflow_count: int
    jobs_count: Optional[int]
    steps_count: Optional[int]
    has_tests: bool
    tests_types: List[str]
    bug_open: int
    bug_closed: int
    commits_recent: int
    prs_merged: int
    workflow_changes: int
    adequacy_score: Optional[float] = None


def run_graphql(query: str, variables: Dict[str, Any], max_retries: int = 5, backoff: float = 2.0) -> Dict[str, Any]:
    attempt = 0
    while True:
        try:
            resp = requests.post(API_GRAPHQL, json={"query": query, "variables": variables}, headers=HEADERS, timeout=30)
            # Treat transient 5xx as retryable
            if resp.status_code >= 500:
                raise requests.exceptions.HTTPError(f"HTTP {resp.status_code}")
            resp.raise_for_status()
            payload = resp.json()
            if payload.get("errors"):
                LOGGER.warning("GraphQL warnings/errors: %s", payload.get("errors"))
            return payload.get("data", {})
        except requests.exceptions.RequestException as exc:
            attempt += 1
            if attempt > max_retries:
                LOGGER.error("GraphQL request failed after %s attempts: %s", attempt - 1, exc)
                raise
            delay = backoff * (2 ** (attempt - 1))
            LOGGER.warning("Falha temporária na API (%s). Tentando novamente em %.1fs... (tentativa %s/%s)", exc, delay, attempt, max_retries)
            time.sleep(delay)


def prepare_query() -> str:
    return """
    query($owner: String!, $name: String!, $since: GitTimestamp!) {
      repository(owner: $owner, name: $name) {
        nameWithOwner
        stargazerCount
        forkCount
        pushedAt
        primaryLanguage { name }
        workflows: object(expression: "HEAD:.github/workflows") {
          ... on Tree { entries { name type object { ... on Blob { text } } } }
        }
        issuesBugOpen: issues(labels: ["bug"], states: OPEN) { totalCount }
        issuesBugClosed: issues(labels: ["bug"], states: CLOSED) { totalCount }
        pullRequests(states: MERGED) { totalCount }
        defaultBranchRef { target { ... on Commit { history(first:0, since: $since) { totalCount } } } }
      }
    }
    """


def analyze_repository(owner: str, name: str, since_iso: str) -> Optional[DeepMetrics]:
    query = prepare_query()
    variables = {"owner": owner, "name": name, "since": since_iso}
    data = run_graphql(query, variables)
    repo = data.get("repository") or {}
    if not repo:
        LOGGER.error("Repositório %s/%s não encontrado", owner, name)
        return None

    # Workflows: count entries, try detectar jobs/steps básico via parsing (heurística)
    workflows = (repo.get("workflows") or {}).get("entries") or []
    workflow_count = len(workflows)
    jobs_count = 0
    steps_count = 0
    has_tests = False
    tests_types: List[str] = []

    for entry in workflows:
        blob = (entry.get("object") or {})
        text = blob.get("text") or ""
        # heurística simples: conta 'jobs:' e 'steps:' aparições
        jobs_count += text.count("jobs:")
        steps_count += text.count("steps:")
        lowered = text.lower()
        if "pytest" in lowered or "unittest" in lowered:
            has_tests = True
            if "pytest" in lowered:
                tests_types.append("unit/pytest")
        if "e2e" in lowered or "cypress" in lowered or "playwright" in lowered:
            has_tests = True
            tests_types.append("e2e")

    bug_open = (repo.get("issuesBugOpen") or {}).get("totalCount", 0)
    bug_closed = (repo.get("issuesBugClosed") or {}).get("totalCount", 0)
    prs_merged = (repo.get("pullRequests") or {}).get("totalCount", 0)
    commits_recent = (repo.get("defaultBranchRef") or {}).get("target", {}).get("history", {}).get("totalCount", 0)

    # Workflow changes: buscar em REST API comparando commits que alteram .github/workflows
    workflow_changes = count_workflow_changes_rest(owner, name)

    assessment = DeepMetrics(
        full_name=f"{owner}/{name}",
        stars=repo.get("stargazerCount", 0),
        forks=repo.get("forkCount", 0),
        pushed_at=repo.get("pushedAt", ""),
        primary_language=(repo.get("primaryLanguage") or {}).get("name"),
        workflow_count=workflow_count,
        jobs_count=jobs_count,
        steps_count=steps_count,
        has_tests=has_tests,
        tests_types=list(set(tests_types)),
        bug_open=bug_open,
        bug_closed=bug_closed,
        commits_recent=commits_recent,
        prs_merged=prs_merged,
        workflow_changes=workflow_changes,
    )

    assessment.adequacy_score = compute_deep_score(assessment)
    return assessment


def count_workflow_changes_rest(owner: str, name: str) -> int:
    """Conta commits no histórico que alteram arquivos em .github/workflows usando REST API.
    Usamos um endpoint simples de commits (paginado) e verificamos modificações nos arquivos.
    Atenção: para repositórios grandes pode ser custoso; heurística limitação a 200 commits recentes.
    """
    url = f"{API_REST}/repos/{owner}/{name}/commits"
    params = {"per_page": 100}
    try:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
        resp.raise_for_status()
        commits = resp.json()
    except Exception:
        return 0

    count = 0
    for c in commits:
        sha = c.get("sha")
        if not sha:
            continue
        # buscar arquivos modificados nesse commit
        try:
            r2 = requests.get(f"{API_REST}/repos/{owner}/{name}/commits/{sha}", headers=HEADERS, timeout=30)
            r2.raise_for_status()
            files = r2.json().get("files", [])
            for f in files:
                filename = f.get("filename", "")
                if filename.startswith(".github/workflows/"):
                    count += 1
                    break
        except Exception:
            continue
    return count


def compute_deep_score(m: DeepMetrics) -> float:
    score = 0.0
    # exemplo de pontuação: peso para workflows, jobs, presence tests e atividade
    score += min(5.0, m.workflow_count) * 0.8
    score += min(10.0, (m.jobs_count or 0)) * 0.1
    score += (2.0 if m.has_tests else 0.0)
    score += min(5.0, m.commits_recent / 10.0) * 0.5
    return round(score, 2)


def export_results(metrics: List[DeepMetrics], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    records = [asdict(m) for m in metrics]
    with (out_dir / "profundo_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    pd.DataFrame(records).to_csv(out_dir / "profundo_metrics.csv", index=False)

    # exemplo de gráfico: score por repositório
    df = pd.DataFrame(records)
    df = df.sort_values(by="adequacy_score", ascending=False)
    plt.figure(figsize=(10, 6))
    plt.bar(df["full_name"], df["adequacy_score"].astype(float))
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Adequacy score (proposto)")
    plt.tight_layout()
    plt.savefig(out_dir / "profundo_scores.png")


def main(repo_list: List[str], window_days: int = 180) -> None:
    since = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()
    metrics: List[DeepMetrics] = []
    for full in repo_list:
        owner, name = full.split("/", 1)
        m = analyze_repository(owner, name, since)
        if m:
            metrics.append(m)

    export_results(metrics, Path("out"))


if __name__ == "__main__":
    # Exemplo: passa lista manualmente
    repos = [
        "huggingface/transformers",
        "pytorch/pytorch",
        "vercel/next.js",
    ]
    main(repos[:10])
