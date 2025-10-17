#!/usr/bin/env python3
"""Curadoria automatizada de repositorios GitHub para estudo de maturidade CI/CD.

Esta implementacao segue a linha metodologica de Valenzuela-Toledo et al. (2024),
Wang et al. (2022) e Veloso & Hora (2022), privilegiando uma analise profunda de
um conjunto reduzido de projetos. O script coleta dados relevantes de pipelines,
testes e issues rotuladas como bug, atribuindo uma pontuacao de adequacao que
apoia a selecao transparente de casos para estudos empiricos replicaveis.
"""
from __future__ import annotations

import csv
import logging
import os
import sys
import time
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# Configuracoes globais -----------------------------------------------------
LOG_LEVEL = os.environ.get("CURADORIA_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="[%(asctime)s] %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger("curadoria")

TARGET_SELECTION_COUNT = int(os.environ.get("CURADORIA_TARGET_COUNT", "10"))
MIN_SCORE_THRESHOLD = int(os.environ.get("CURADORIA_MIN_SCORE", "6"))
MIN_WORKFLOW_COUNT = int(os.environ.get("CURADORIA_MIN_WORKFLOWS", "30"))
SEARCH_QUERY = os.environ.get("CURADORIA_SEARCH_QUERY", "stars:>10000 sort:stars-desc")
SEARCH_PAGE_SIZE = int(os.environ.get("CURADORIA_SEARCH_PAGE_SIZE", "10"))
MAX_SEARCH_PAGES = int(os.environ.get("CURADORIA_MAX_SEARCH_PAGES", "120"))
AUTO_DISCOVERY_ENABLED = os.environ.get("CURADORIA_AUTO_DISCOVERY", "true").lower() != "false"
SEARCH_MAX_RETRIES = int(os.environ.get("CURADORIA_MAX_RETRIES", "3"))
SEARCH_RETRY_BACKOFF = float(os.environ.get("CURADORIA_RETRY_BACKOFF", "2.0"))

# Lista manual de repositorios para fallback, caso a descoberta automatica nao atinja a meta.
REPOSITORIES = [
    "facebook/react",
    "axios/axios",
    "pandas-dev/pandas",
    "vercel/next.js",
    "tiangolo/fastapi",
    "microsoft/typescript",
]

# Palavras-chave que indicam execucao de testes automatizados.
TEST_COMMAND_KEYWORDS = (
    "pytest",
    "npm test",
    "yarn test",
    "pnpm test",
    "mvn test",
    "mvn -q test",
    "gradle test",
    "./gradlew test",
    "go test",
    "cargo test",
    "dotnet test",
    "pytest -",
    "tox",
)

RATE_LIMIT_BUFFER = 25  # mantemos folga para evitar atingir o limite acidentalmente
RECENT_ACTIVITY_DAYS = 180
DEFAULT_ENV_FILE = Path(__file__).resolve().parent / ".env"
OUT_DIR = Path("out")
OUT_JSON = OUT_DIR / "curadoria_resultados.json"
OUT_CSV = OUT_DIR / "curadoria_resultados.csv"
OUT_JUST_MD = OUT_DIR / "justificativas.md"
OUT_JUST_TEX = OUT_DIR / "justificativas.tex"


@dataclass
class RepositoryAssessment:
    """Agrega os dados coletados para facilitar exportacoes e relatorios."""

    full_name: str
    stars: int
    forks: int
    pushed_at: str
    primary_language: Optional[str]
    workflow_count: int
    has_test_commands: bool
    has_test_directory: bool
    bug_issues_open: int
    bug_issues_closed: int
    adequacy_score: int
    workflow_files: List[str] = field(default_factory=list)

    @property
    def bug_issues_total(self) -> int:
        return self.bug_issues_open + self.bug_issues_closed


# Funcoes utilitarias -------------------------------------------------------

def load_token(env_path: Path) -> str:
    """Le o token GITHUB_TOKEN do arquivo .env."""
    if not env_path.exists():
        LOGGER.error("Arquivo .env nao encontrado em %s", env_path)
        raise FileNotFoundError("Arquivo .env ausente")

    token: Optional[str] = None
    with env_path.open(encoding="utf-8") as handler:
        for line in handler:
            if line.strip().startswith("#") or "=" not in line:
                continue
            key, value = line.strip().split("=", 1)
            if key == "GITHUB_TOKEN":
                token = value.strip().strip('"').strip("'")
                break

    if not token:
        LOGGER.error("GITHUB_TOKEN nao definido no .env")
        raise ValueError("Token GitHub ausente no arquivo .env")

    return token


def run_graphql(token: str, query: str, variables: Dict[str, Any]) -> Dict[str, Any]:
    """Executa uma consulta GraphQL com tratamento basico de erros."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    response = requests.post(
        "https://api.github.com/graphql",
        json={"query": query, "variables": variables},
        headers=headers,
        timeout=30,
    )

    if response.status_code == 401:
        LOGGER.error("Falha de autenticacao com o token fornecido.")
        raise PermissionError("Token invalido ou sem escopo adequado.")

    if response.status_code == 403 and response.headers.get("X-RateLimit-Remaining") == "0":
        reset_at = response.headers.get("X-RateLimit-Reset")
        LOGGER.error("Limite de requisicoes excedido. Reset em %s", reset_at)
        raise RuntimeError("Rate limit excedido para a API GraphQL do GitHub.")

    if 500 <= response.status_code < 600:
        truncated = response.text[:200].replace("\n", " ") if response.text else ""
        LOGGER.warning(
            "Erro temporario na API (HTTP %s): %s",
            response.status_code,
            truncated,
        )
        raise RuntimeError(f"Erro temporario HTTP {response.status_code}")

    if not response.ok:
        LOGGER.error("Erro HTTP %s: %s", response.status_code, response.text)
        response.raise_for_status()

    payload = response.json()
    if "errors" in payload:
        messages = ", ".join(err.get("message", "Erro desconhecido") for err in payload["errors"])
        if any("rate limit" in msg.lower() for msg in messages.split(",")):
            LOGGER.error("Erros de rate limit: %s", messages)
            raise RuntimeError("Rate limit excedido segundo resposta GraphQL.")
        raise RuntimeError(f"Erro na consulta GraphQL: {messages}")

    return payload


def detect_test_commands(workflow_entries: List[Dict[str, Any]]) -> bool:
    """Detecta comandos tipicos de execucao de testes em arquivos YAML."""
    for entry in workflow_entries:
        blob = entry.get("object") or {}
        text = blob.get("text") or ""
        lowered = text.lower()
        if any(keyword in lowered for keyword in TEST_COMMAND_KEYWORDS):
            return True
    return False


def compute_adequacy_score(assessment: RepositoryAssessment) -> int:
    """Aplica as regras de pontuacao definidas no protocolo do estudo."""
    score = 0
    if assessment.workflow_count > 0:
        score += 2
    if assessment.has_test_commands:
        score += 2
    if assessment.has_test_directory:
        score += 1
    if assessment.stars > 10_000:
        score += 1
    if assessment.forks > 1_000:
        score += 1
    if assessment.bug_issues_total >= 50:
        score += 1

    pushed_datetime = datetime.fromisoformat(assessment.pushed_at.replace("Z", "+00:00"))
    if datetime.now(timezone.utc) - pushed_datetime <= timedelta(days=RECENT_ACTIVITY_DAYS):
        score += 1

    if assessment.primary_language:
        score += 1

    return score


def build_assessment_from_repository_data(
    repository: Dict[str, Any],
    fallback_full_name: str,
) -> Optional[RepositoryAssessment]:
    """Transforma o payload GraphQL em RepositoryAssessment."""

    if repository is None:
        return None

    workflows_tree = repository.get("workflows") or {}
    workflow_entries = workflows_tree.get("entries") or []
    workflow_files = [entry.get("name", "") for entry in workflow_entries]
    has_test_commands = detect_test_commands(workflow_entries)

    has_test_directory = any(
        repository.get(alias) is not None
        for alias in ("directoryTests", "directoryTest", "directoryUnderscoreTests")
    )

    assessment = RepositoryAssessment(
        full_name=repository.get("nameWithOwner", fallback_full_name),
        stars=repository.get("stargazerCount", 0),
        forks=repository.get("forkCount", 0),
        pushed_at=repository.get("pushedAt", ""),
        primary_language=(repository.get("primaryLanguage") or {}).get("name"),
        workflow_count=len(workflow_entries),
        has_test_commands=has_test_commands,
        has_test_directory=has_test_directory,
        bug_issues_open=(repository.get("issuesBugOpen") or {}).get("totalCount", 0),
        bug_issues_closed=(repository.get("issuesBugClosed") or {}).get("totalCount", 0),
        adequacy_score=0,
        workflow_files=workflow_files,
    )

    assessment.adequacy_score = compute_adequacy_score(assessment)
    return assessment


def prepare_query() -> str:
    """Retorna a query GraphQL utilizada para coletar os dados."""
    return """
    query($owner: String!, $name: String!) {
      repository(owner: $owner, name: $name) {
        nameWithOwner
        stargazerCount
        forkCount
        pushedAt
        primaryLanguage { name }
        workflows: object(expression: "HEAD:.github/workflows") {
          ... on Tree {
            entries {
              name
              type
              object {
                ... on Blob {
                  text
                }
              }
            }
          }
        }
        directoryTests: object(expression: "HEAD:tests") { id }
        directoryTest: object(expression: "HEAD:test") { id }
        directoryUnderscoreTests: object(expression: "HEAD:__tests__") { id }
        issuesBugOpen: issues(labels: ["bug"], states: OPEN) { totalCount }
        issuesBugClosed: issues(labels: ["bug"], states: CLOSED) { totalCount }
      }
      rateLimit {
        remaining
        resetAt
        cost
      }
    }
    """


def prepare_search_query() -> str:
        """Retorna a query GraphQL para descoberta automatica via Search API."""
        return """
        query($query: String!, $first: Int!, $after: String) {
            search(query: $query, type: REPOSITORY, first: $first, after: $after) {
                repositoryCount
                pageInfo {
                    hasNextPage
                    endCursor
                }
                edges {
                    node {
                        ... on Repository {
                            nameWithOwner
                            stargazerCount
                            forkCount
                            pushedAt
                            primaryLanguage { name }
                            workflows: object(expression: "HEAD:.github/workflows") {
                                ... on Tree {
                                    entries {
                                        name
                                        type
                                        object {
                                            ... on Blob {
                                                text
                                            }
                                        }
                                    }
                                }
                            }
                            directoryTests: object(expression: "HEAD:tests") { id }
                            directoryTest: object(expression: "HEAD:test") { id }
                            directoryUnderscoreTests: object(expression: "HEAD:__tests__") { id }
                            issuesBugOpen: issues(labels: ["bug"], states: OPEN) { totalCount }
                            issuesBugClosed: issues(labels: ["bug"], states: CLOSED) { totalCount }
                        }
                    }
                }
            }
            rateLimit {
                remaining
                resetAt
                cost
            }
        }
        """


def analyze_repository(token: str, full_name: str, query: str) -> Optional[RepositoryAssessment]:
    """Coleta e calcula os dados para um repositorio especifico."""
    owner, name = full_name.split("/", maxsplit=1)

    try:
        result = run_graphql(token, query, {"owner": owner, "name": name})
    except RuntimeError as exc:
        LOGGER.error("Erro ao consultar %s: %s", full_name, exc)
        return None
    except PermissionError as exc:
        LOGGER.error("Permissao insuficiente para %s: %s", full_name, exc)
        return None

    rate_info = (result.get("data") or {}).get("rateLimit") or {}
    remaining = rate_info.get("remaining")
    if remaining is not None and remaining < RATE_LIMIT_BUFFER:
        LOGGER.warning(
            "Rate limit baixo (%s requisicoes restantes). Considere pausar ou reduzir o escopo.",
            remaining,
        )

    repository = (result.get("data") or {}).get("repository")
    if repository is None:
        LOGGER.error("Repositorio %s nao encontrado ou inacessivel.", full_name)
        return None

    assessment = build_assessment_from_repository_data(repository, full_name)
    if assessment is None:
        LOGGER.error("Payload inesperado para %s", full_name)
        return None
    LOGGER.info(
        "Repositorio %s analisado: score=%s, workflows=%s, testes=%s, bugs=%s",
        assessment.full_name,
        assessment.adequacy_score,
        assessment.workflow_count,
        assessment.has_test_commands,
        assessment.bug_issues_total,
    )
    return assessment


def discover_repositories(token: str) -> List[RepositoryAssessment]:
    """Busca repositorios populares e retorna os que atingirem a pontuacao minima."""

    if not AUTO_DISCOVERY_ENABLED:
        LOGGER.info("Descoberta automatica desativada por configuracao.")
        return []

    query = prepare_search_query()
    after_cursor: Optional[str] = None
    selected: List[RepositoryAssessment] = []
    seen: set[str] = set()
    pages = 0

    per_page = max(1, min(SEARCH_PAGE_SIZE, 100))
    desired = max(1, TARGET_SELECTION_COUNT)

    LOGGER.info(
        "Iniciando descoberta automatica: query='%s', alvo=%s, pagina=%s registros",
        SEARCH_QUERY,
        desired,
        per_page,
    )

    while len(selected) < desired and pages < MAX_SEARCH_PAGES:
        variables = {"query": SEARCH_QUERY, "first": per_page, "after": after_cursor}
        payload: Optional[Dict[str, Any]] = None
        attempt = 0
        while attempt < SEARCH_MAX_RETRIES:
            try:
                payload = run_graphql(token, query, variables)
                break
            except RuntimeError as exc:
                attempt += 1
                if attempt >= SEARCH_MAX_RETRIES:
                    LOGGER.error("Erro ao executar busca GraphQL apos %s tentativas: %s", attempt, exc)
                    break
                delay = SEARCH_RETRY_BACKOFF * attempt
                LOGGER.warning("Falha temporaria na busca (%s). Tentando novamente em %.1fs...", exc, delay)
                time.sleep(delay)
            except PermissionError as exc:
                LOGGER.error("Permissao insuficiente na busca: %s", exc)
                return selected

        if payload is None:
            LOGGER.warning("Busca interrompida por erros recorrentes.")
            break

        data = payload.get("data") or {}
        search_block = data.get("search") or {}
        edges = search_block.get("edges") or []

        if not edges:
            LOGGER.info("Sem resultados adicionais na busca (pagina %s).", pages + 1)
            break

        for edge in edges:
            repo_node = edge.get("node") or {}
            if not repo_node:
                continue

            full_name = repo_node.get("nameWithOwner")
            if not full_name or full_name in seen:
                continue

            assessment = build_assessment_from_repository_data(repo_node, full_name or "")
            if assessment is None:
                continue

            seen.add(assessment.full_name)

            if not assessment.has_test_commands:
                LOGGER.debug(
                    "Ignorando %s: nenhum comando de teste detectado nos workflows.",
                    assessment.full_name,
                )
                continue

            if assessment.workflow_count < MIN_WORKFLOW_COUNT:
                LOGGER.debug(
                    "Ignorando %s: apenas %s workflows (minimo exigido: %s).",
                    assessment.full_name,
                    assessment.workflow_count,
                    MIN_WORKFLOW_COUNT,
                )
                continue

            if assessment.adequacy_score >= MIN_SCORE_THRESHOLD:
                selected.append(assessment)
                LOGGER.info(
                    "Selecionado %s (score=%s, workflows=%s)",
                    assessment.full_name,
                    assessment.adequacy_score,
                    assessment.workflow_count,
                )
                if len(selected) >= desired:
                    break
            else:
                LOGGER.debug(
                    "Ignorando %s: score %s abaixo do limiar %s.",
                    assessment.full_name,
                    assessment.adequacy_score,
                    MIN_SCORE_THRESHOLD,
                )

        page_info = search_block.get("pageInfo") or {}
        if len(selected) >= desired:
            break
        if not page_info.get("hasNextPage"):
            LOGGER.info("Busca alcancou o fim dos resultados (pagina %s).", pages + 1)
            break

        after_cursor = page_info.get("endCursor")
        pages += 1

        rate_info = data.get("rateLimit") or {}
        remaining = rate_info.get("remaining")
        if remaining is not None and remaining < RATE_LIMIT_BUFFER:
            LOGGER.warning(
                "Rate limit baixo (%s requisicoes restantes) apos descoberta. Encerrando ciclo.",
                remaining,
            )
            break

    LOGGER.info(
        "Descoberta automatica retornou %s repositorios qualificados (paginas percorridas: %s).",
        len(selected),
        pages + 1 if pages or selected else 0,
    )
    return selected


def build_outputs(assessments: List[RepositoryAssessment]) -> None:
    """Gera arquivos de saida (JSON, CSV e justificativas)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ordered = sorted(
        assessments,
        key=lambda item: (-item.adequacy_score, -item.stars, -item.workflow_count),
    )

    data_for_export = [
        {
            "repository": item.full_name,
            "owner": item.full_name.split("/")[0],
            "name": item.full_name.split("/")[1],
            "adequacy_score": item.adequacy_score,
            "stars": item.stars,
            "forks": item.forks,
            "pushed_at": item.pushed_at,
            "primary_language": item.primary_language,
            "workflow_count": item.workflow_count,
            "has_test_commands_in_workflows": item.has_test_commands,
            "has_test_directory": item.has_test_directory,
            "bug_issues_total": item.bug_issues_total,
            "bug_issues_open": item.bug_issues_open,
            "bug_issues_closed": item.bug_issues_closed,
            "workflow_files": item.workflow_files,
        }
        for item in ordered
    ]

    with OUT_JSON.open("w", encoding="utf-8") as handler:
        json.dump(data_for_export, handler, ensure_ascii=False, indent=2)
    LOGGER.info("Resultados salvos em %s", OUT_JSON)

    csv_headers = list(data_for_export[0].keys()) if data_for_export else []
    if csv_headers:
        with OUT_CSV.open("w", encoding="utf-8", newline="") as handler:
            writer = csv.DictWriter(handler, fieldnames=csv_headers)
            writer.writeheader()
            writer.writerows(data_for_export)
        LOGGER.info("Resultados salvos em %s", OUT_CSV)

    finalists = [item for item in ordered if item.adequacy_score >= MIN_SCORE_THRESHOLD]

    markdown_lines: List[str] = [
    "# Justificativas de Selecao\n",
    ]
    latex_lines: List[str] = []

    for item in finalists:
        markdown_lines.extend([
            f"**{item.full_name}**  ",
            "Selecionado por: (i) uso ativo de GitHub Actions "
            f"(workflows >= {item.workflow_count});  ",
            "(ii) presenca de testes automatizados;  ",
            "(iii) atividade continua "
            f"(stars >= {item.stars}, forks >= {item.forks});  ",
            "(iv) politica consistente de labels de bug "
            f"(bug_labels_total >= {item.bug_issues_total}).\n\n",
        ])

        latex_lines.append(
            "\\paragraph{" + item.full_name + ".} Seleciona-se este repositorio por atender aos criterios de elegibilidade: "
            "(i) uso ativo do \\textit{GitHub Actions}; (ii) execucao de testes automatizados nos \\textit{pipelines}; "
            "(iii) atividade continua da comunidade; (iv) politica consistente de rotulagem de \\textit{issues} de \\textit{bug}. "
            "Esses fatores indicam maturidade operacional e disponibilidade de dados para a analise longitudinal proposta.\n"
        )

    with OUT_JUST_MD.open("w", encoding="utf-8") as handler:
        handler.writelines(markdown_lines)
    LOGGER.info("Justificativas Markdown salvas em %s", OUT_JUST_MD)

    with OUT_JUST_TEX.open("w", encoding="utf-8") as handler:
        handler.writelines(latex_lines)
    LOGGER.info("Justificativas LaTeX salvas em %s", OUT_JUST_TEX)

    print("\nResumo da Curadoria:\n")
    for item in ordered:
        lang = item.primary_language or "Indefinida"
        tests = "sim" if item.has_test_commands else "nao"
        print(
            f"- {item.full_name} | Linguagem: {lang} | Workflows: {item.workflow_count} | "
            f"Testes no CI: {tests} | Score: {item.adequacy_score}"
        )


def main() -> None:
    try:
        token = load_token(DEFAULT_ENV_FILE)
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.critical("Nao foi possivel carregar o token: %s", exc)
        sys.exit(1)

    assessments: List[RepositoryAssessment] = []
    selected_names: set[str] = set()

    discovered = discover_repositories(token)
    assessments.extend(discovered)
    selected_names.update(item.full_name for item in discovered)

    remaining_slots = max(0, TARGET_SELECTION_COUNT - len(assessments))

    if remaining_slots and REPOSITORIES:
        LOGGER.info(
            "Complementando com lista manual para preencher %s repositorios restantes.",
            remaining_slots,
        )
        manual_query = prepare_query()
        for full_name in REPOSITORIES:
            if full_name in selected_names:
                continue
            LOGGER.info("Processando fallback %s", full_name)
            assessment = analyze_repository(token, full_name, manual_query)
            if not assessment:
                continue
            if not assessment.has_test_commands:
                LOGGER.debug(
                    "Ignorando fallback %s: nenhum comando de teste nos workflows.",
                    assessment.full_name,
                )
                continue
            if assessment.workflow_count < MIN_WORKFLOW_COUNT:
                LOGGER.debug(
                    "Ignorando fallback %s: apenas %s workflows (minimo exigido: %s).",
                    assessment.full_name,
                    assessment.workflow_count,
                    MIN_WORKFLOW_COUNT,
                )
                continue
            if assessment.adequacy_score >= MIN_SCORE_THRESHOLD:
                assessments.append(assessment)
                selected_names.add(assessment.full_name)
            else:
                LOGGER.debug(
                    "Ignorando fallback %s: score %s abaixo do limiar %s.",
                    assessment.full_name,
                    assessment.adequacy_score,
                    MIN_SCORE_THRESHOLD,
                )
            if len(assessments) >= TARGET_SELECTION_COUNT:
                break

    if not assessments:
        LOGGER.error("Nenhum repositorio qualificado foi encontrado. Encerrando.")
        sys.exit(1)

    if len(assessments) < TARGET_SELECTION_COUNT:
        LOGGER.warning(
            "Somente %s repositorios atingiram a pontuacao minima (meta=%s).",
            len(assessments),
            TARGET_SELECTION_COUNT,
        )

    build_outputs(assessments)


if __name__ == "__main__":
    main()
