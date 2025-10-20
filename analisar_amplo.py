"""Análise Ampla (100-1000 repositórios)

Coleta indicadores sintéticos e calcula índices simplificados (PMI, TMI) para muitos repositórios.
Gera CSV com PMI, TMI, taxa de bugs e permite análises de correlação.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from scipy import stats
from sklearn.preprocessing import minmax_scale

from dotenv import load_dotenv

load_dotenv()

LOGGER = logging.getLogger("analisar_amplo")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    LOGGER.critical("GITHUB_TOKEN não encontrado no .env")
    raise SystemExit(1)

HEADERS = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"}
API_GRAPHQL = "https://api.github.com/graphql"
API_REST = "https://api.github.com"


@dataclass
class WideMetrics:
    full_name: str
    stars: int
    has_workflows: bool
    has_tests: bool
    commits_recent: int
    bug_rate: float
    pmi: Optional[float] = None
    tmi: Optional[float] = None


def run_graphql(query: str, variables: Dict[str, Any], max_retries: int = 6, base_backoff: float = 2.0, jitter: float = 0.5) -> Dict[str, Any]:
    attempt = 0
    while True:
        try:
            resp = requests.post(API_GRAPHQL, json={"query": query, "variables": variables}, headers=HEADERS, timeout=30)
            if resp.status_code >= 500:
                raise requests.exceptions.HTTPError(f"HTTP {resp.status_code}")
            resp.raise_for_status()
            payload = resp.json()
            return payload.get("data", {})
        except requests.exceptions.RequestException as exc:
            attempt += 1
            if attempt > max_retries:
                LOGGER.error("GraphQL request failed after %s attempts: %s", attempt - 1, exc)
                raise
            # exponential backoff with jitter
            delay = base_backoff * (2 ** (attempt - 1))
            delay = delay + random.uniform(0, jitter)
            LOGGER.warning("Falha temporária na API (%s). Tentando novamente em %.1fs... (tentativa %s/%s)", exc, delay, attempt, max_retries)
            time.sleep(delay)


def prepare_search_query() -> str:
    # Retorna campos mínimos: name, stars, pushedAt, workflows tree existence, issues bug counts
    return """
    query($query: String!, $first: Int!, $after: String) {
      search(query: $query, type: REPOSITORY, first: $first, after: $after) {
        edges { node { ... on Repository {
            nameWithOwner
            stargazerCount
            pushedAt
            workflows: object(expression: "HEAD:.github/workflows") { ... on Tree { entries { name } } }
            issuesBugOpen: issues(labels: ["bug"], states: OPEN) { totalCount }
            issuesBugClosed: issues(labels: ["bug"], states: CLOSED) { totalCount }
        } } }
        pageInfo { hasNextPage endCursor }
      }
      rateLimit { remaining resetAt cost }
    }
    """


def estimate_indicators_from_node(node: Dict[str, Any]) -> WideMetrics:
    full = node.get("nameWithOwner")
    stars = node.get("stargazerCount", 0)
    workflows = node.get("workflows") or {}
    has_workflows = bool((workflows.get("entries") or []))
    bug_open = (node.get("issuesBugOpen") or {}).get("totalCount", 0)
    bug_closed = (node.get("issuesBugClosed") or {}).get("totalCount", 0)
    bug_rate = bug_open / max(1, (bug_open + bug_closed))

    # commits_recent via REST quick endpoint (cheap: only get default branch commit count approximation)
    commits_recent = estimate_commits_quick(full)

    # heurística de detecção de testes: se workflows existirem e mencionam pytest/npm test
    has_tests = False
    entries = (workflows.get("entries") or [])
    for e in entries:
        if "test" in (e.get("name") or "").lower():
            has_tests = True

    return WideMetrics(full_name=full, stars=stars, has_workflows=has_workflows, has_tests=has_tests, commits_recent=commits_recent, bug_rate=bug_rate)


def estimate_commits_quick(full_name: str) -> int:
    # pega info via REST para obter commit count do default branch (approx: usar /repos/:owner/:repo)
    try:
        owner, name = full_name.split("/", 1)
        r = requests.get(f"{API_REST}/repos/{owner}/{name}", headers=HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json()
        return data.get("size", 0)  # proxy barato: repository size (não ideal mas rápido)
    except Exception:
        return 0


def compute_pmi_tmi(df: pd.DataFrame) -> pd.DataFrame:
    # PMI: combina has_workflows (binary), commits_recent (scaled) e presence of CI tests
    df = df.copy()
    df["commits_scaled"] = minmax_scale(df["commits_recent"].replace(0, np.nan).fillna(0))
    df["pmi_raw"] = (df["has_workflows"].astype(int) * 0.5) + (df["commits_scaled"] * 0.3) + (df["has_tests"].astype(int) * 0.2)
    df["pmi"] = (df["pmi_raw"] * 10).round(2)

    # TMI: presença de testes e taxa de sucesso inferida via heurística (aqui simplificada)
    df["tmi_raw"] = (df["has_tests"].astype(int) * 0.7) + (df["commits_scaled"] * 0.3)
    df["tmi"] = (df["tmi_raw"] * 10).round(2)

    return df


def run_search_and_collect(search_query: str, max_repos: int = 500, per_page: int = 50) -> pd.DataFrame:
    query = prepare_search_query()
    after = None
    collected: List[WideMetrics] = []
    remaining_slots = max_repos

    while remaining_slots > 0:
        first = min(per_page, remaining_slots)
        variables = {"query": search_query, "first": first, "after": after}
        try:
            data = run_graphql(query, variables)
        except requests.exceptions.RequestException:
            LOGGER.error("Busca GraphQL falhou repetidamente. Tentando fallback para 'out/curadoria_resultados.json'...")
            # fallback: carregar curadoria_resultados.json se existir e extrair nomes
            fallback_path = Path("out/curadoria_resultados.json")
            if fallback_path.exists():
                try:
                    payload = json.loads(fallback_path.read_text(encoding="utf-8"))
                    # payload é lista de objetos (dicionários) com informações geradas por curadoria_repos
                    repos = payload[:max_repos]
                    for item in repos:
                        if not isinstance(item, dict):
                            continue
                        full = item.get("repository") or item.get("nameWithOwner")
                        node = {
                            "nameWithOwner": full,
                            "stargazerCount": item.get("stars", 0),
                            "pushedAt": item.get("pushed_at", None),
                            "workflows": {"entries": item.get("workflow_files", [])},
                            "issuesBugOpen": {"totalCount": item.get("bug_issues_open", 0)},
                            "issuesBugClosed": {"totalCount": item.get("bug_issues_closed", 0)},
                        }
                        collected.append(estimate_indicators_from_node(node))
                    df = pd.DataFrame([asdict(x) for x in collected])
                    df_filled = compute_pmi_tmi(df)
                    return df_filled
                except Exception as exc:
                    LOGGER.error("Falha ao carregar fallback: %s", exc)
                    raise
            else:
                LOGGER.error("Nenhum fallback disponível; abortando.")
                raise
        search = data.get("search") or {}
        edges = search.get("edges") or []
        for edge in edges:
            node = (edge.get("node") or {})
            metrics = estimate_indicators_from_node(node)
            collected.append(metrics)
            remaining_slots -= 1
            if remaining_slots <= 0:
                break
        page_info = search.get("pageInfo") or {}
        if not page_info.get("hasNextPage"):
            break
        after = page_info.get("endCursor")

    df = pd.DataFrame([asdict(x) for x in collected])
    df_filled = compute_pmi_tmi(df)
    return df_filled


def analyze_correlations(df: pd.DataFrame) -> Dict[str, Any]:
    res: Dict[str, Any] = {}
    if df.empty:
        return res
    res["pearson_pmi_bug"] = stats.pearsonr(df["pmi"], df["bug_rate"]) if len(df) > 2 else (None, None)
    res["spearman_pmi_bug"] = stats.spearmanr(df["pmi"], df["bug_rate"]) if len(df) > 2 else (None, None)
    res["pearson_tmi_bug"] = stats.pearsonr(df["tmi"], df["bug_rate"]) if len(df) > 2 else (None, None)
    res["spearman_tmi_bug"] = stats.spearmanr(df["tmi"], df["bug_rate"]) if len(df) > 2 else (None, None)
    return res


def export_wide(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "amplo_metrics.csv", index=False)


def main(search_query: str, max_repos: int = 200) -> None:
    df = run_search_and_collect(search_query, max_repos=max_repos)
    corrs = analyze_correlations(df)
    export_wide(df, Path("out"))
    # salvar resultados das correlações
    with (Path("out") / "amplo_correlations.json").open("w", encoding="utf-8") as f:
        json.dump({k: (None if v is None else (v[0], v[1])) for k, v in corrs.items()}, f, default=str, indent=2)


if __name__ == "__main__":
    # exemplo de execução: stars>50 updated:>2024-01-01
    search_q = "stars:>50 pushed:>2024-01-01 sort:stars-desc"
    main(search_q, max_repos=200)
