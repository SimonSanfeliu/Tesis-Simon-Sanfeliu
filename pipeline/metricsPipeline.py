import json
import os
import shutil
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime
from typing import Any, Optional

import pandas as pd
import requests
import sqlalchemy as sa

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from logger_setup import setup_logger
from prompts.correction.SelfCorrection import (
    general_context_selfcorr_v1,
    general_context_selfcorr_v1_python,
    prompt_self_correction_v2,
)
from secret.config import (
    ANTHROPIC_KEY,
    GOOGLE_KEY,
    OPENAI_KEY,
    PASS_10,
    SQL_URL,
    USER_10,
)

logger = setup_logger(name="metrics", log_file="logs/metrics.txt")

_SQL_PARAMS_CACHE: Optional[dict[str, Any]] = None


def _get_sql_params() -> dict[str, Any]:
    """Load database credentials lazily to avoid import-time side effects."""
    global _SQL_PARAMS_CACHE
    if _SQL_PARAMS_CACHE is None:
        response = requests.get(SQL_URL, timeout=30)
        response.raise_for_status()
        payload = response.json()
        params = payload.get("params")
        if not isinstance(params, dict):
            raise ValueError("SQL params were not found in the configuration payload.")
        _SQL_PARAMS_CACHE = params
    return _SQL_PARAMS_CACHE


def _get_model_provider(model: str) -> str:
    model = model.lower().strip()

    if model.startswith("gpt-") or "codex" in model or model.startswith("o1"):
        return "openai"
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith("gemini-"):
        return "google"

    raise Exception(f"No valid model: {model}")


def _extract_responses_api_text(payload: dict[str, Any]) -> str:
    output_items = payload.get("output", [])
    text_chunks = []

    for item in output_items:
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                text_chunks.append(content.get("text", ""))

    if text_chunks:
        return "".join(text_chunks).strip()

    if payload.get("output_text"):
        return str(payload["output_text"]).strip()

    raise ValueError("Responses API payload did not contain output text.")


def _responses_api_call(model: str, max_tokens: int, prompt: str) -> tuple[str, dict[str, Any]]:
    payload = {
        "model": model,
        "input": prompt,
        "max_output_tokens": max_tokens,
    }
    response = requests.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {OPENAI_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=180,
    )
    response.raise_for_status()
    payload = response.json()
    usage_payload = payload.get("usage", {})
    usage = {
        "input_tokens": usage_payload.get("input_tokens", 0),
        "output_tokens": usage_payload.get("output_tokens", 0),
        "total_tokens": usage_payload.get("total_tokens", 0),
    }
    return _extract_responses_api_text(payload), usage


def _api_call(model: str, max_tokens: int, prompt: str) -> tuple[str, dict[str, Any]]:
    provider = _get_model_provider(model)
    model_lower = model.lower().strip()

    if provider == "openai":
        import openai

        if model_lower == "gpt-5.2-codex":
            return _responses_api_call(model, max_tokens, prompt)

        client = openai.OpenAI(api_key=OPENAI_KEY)
        request_kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }

        if model_lower.startswith("o1") or model_lower.startswith("gpt-5") or "codex" in model_lower:
            request_kwargs["max_completion_tokens"] = max_tokens
        else:
            request_kwargs["temperature"] = 0
            request_kwargs["max_tokens"] = max_tokens

        response = client.chat.completions.create(**request_kwargs)
        usage = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        return response.choices[0].message.content, usage

    if provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        response = client.messages.create(
            model=model,
            temperature=0,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        usage = response.usage.to_dict()
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
        return response.content[0].text, usage

    if provider == "google":
        from google import genai

        client = genai.Client(api_key=GOOGLE_KEY)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={"temperature": 0, "max_output_tokens": max_tokens},
        )
        usage = {
            "input_tokens": response.usage_metadata.prompt_token_count,
            "output_tokens": response.usage_metadata.candidates_token_count,
            "total_tokens": response.usage_metadata.total_token_count,
        }
        return response.text, usage

    raise Exception(f"No valid provider for model: {model}")


def _format_response(specified_format: str, response: str) -> str:
    if specified_format == "sql":
        formatted_response = response.split("```sql")[1].split("```")[0]
        formatted_response = formatted_response.replace("```", "").replace("```sql", "")
    elif specified_format == "python":
        formatted_response = response.split("```python")[1].split("```")[0]
        formatted_response = formatted_response.replace("```", "").replace("```python", "")
        formatted_response = formatted_response.replace('"""""', '"""')
    else:
        raise Exception("No valid format specified")

    return formatted_response.replace(";", "")


def _choose_id_column(df):
    """Pick a stable row-identifier column if present; else None."""
    candidates = ["oid", "oid_catalog", "objectidps1", "classifier_name", "count"]
    for c in candidates:
        if c in df.columns:
            return c

    lower = {c.lower(): c for c in df.columns}
    for alt in ["ztf_identifier", "ztf identifier", "ztf_oid", "object", "ztf"]:
        if alt in lower:
            return lower[alt]
    return None


def compute_row_metrics(query_pred, query_gold):
    """
    Returns (r_row, p_row, N_perfect_row) with correct counting (no >1).
    Uses multiset intersection if duplicates appear.
    """
    if query_pred is None or query_gold is None:
        return 0.0, 0.0, 0

    query_pred = query_pred.loc[:, ~query_pred.columns.duplicated()]
    query_gold = query_gold.loc[:, ~query_gold.columns.duplicated()]

    id_col = _choose_id_column(query_pred)
    if id_col is None or id_col not in query_gold.columns:
        return 0.0, 0.0, 0

    pred_ids = (
        query_pred.sort_values(by=id_col, axis=0).reset_index(drop=True)[id_col].tolist()
    )
    gold_ids = (
        query_gold.sort_values(by=id_col, axis=0).reset_index(drop=True)[id_col].tolist()
    )

    if pred_ids == gold_ids:
        return 1.0, 1.0, 1

    c_pred = Counter(pred_ids)
    c_gold = Counter(gold_ids)
    tp = sum(min(c_pred[k], c_gold[k]) for k in c_pred.keys() & c_gold.keys())

    n_pred = len(pred_ids)
    n_gold = len(gold_ids)

    r_row = 0.0 if n_gold == 0 else tp / n_gold
    p_row = 0.0 if n_pred == 0 else tp / n_pred
    n_perfect_row = 1 if r_row == 1.0 and p_row == 1.0 else 0
    return float(r_row), float(p_row), int(n_perfect_row)


class metricsPipeline:
    METRICS_COLUMNS = [
        "code_tag",
        "llm_used",
        "prompt_version",
        "query_id",
        "query_run",
        "sql_query",
        "tab_schema",
        "label",
        "query_gen_time",
        "query_gen_date",
        "request_text",
        "query_results",
        "query_error",
        "sql_time",
        "sql_date",
        "r_row",
        "p_row",
        "r_col",
        "p_col",
        "N_perfect_row",
        "N_perfect_col",
    ]

    def __init__(
        self,
        llm,
        lang_type,
        max_tokens,
        t_conn,
        n_tries,
        direct,
        self_corr,
        self_corr_prompts,
        prompts_path,
    ):
        """
        Metrics pipeline class
        """
        self.llm = llm
        self.original_lang_type = lang_type
        self.lang_type = lang_type
        self.max_tokens = max_tokens
        self.t_conn = t_conn
        self.n_tries = n_tries
        self.direct = direct
        self.self_corr = self_corr
        self.self_corr_prompts = self_corr_prompts
        self.prompts_path = prompts_path

        self.new_df = None

    def _to_query_key(self, value: Any) -> Optional[str]:
        if value is None:
            return None

        text = str(value).strip()
        if not text or text.lower() in {"nan", "none"}:
            return None

        try:
            numeric = float(text)
        except ValueError:
            return text

        if numeric.is_integer():
            return str(int(numeric))
        return text

    def _to_run_number(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        numeric = pd.to_numeric(value, errors="coerce")
        if pd.isna(numeric):
            return None
        numeric = float(numeric)
        if not numeric.is_integer():
            return None
        return int(numeric)

    def _is_missing(self, value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return value.strip() == "" or value.strip().lower() in {"nan", "none"}
        try:
            return bool(pd.isna(value))
        except Exception:
            return False

    def _coalesce(self, *values: Any) -> Any:
        for value in values:
            if not self._is_missing(value):
                return value
        return None

    def _stringify_error(self, error: Any) -> Optional[str]:
        if self._is_missing(error):
            return None
        return str(error)

    def _merge_errors(self, *errors: Any) -> Optional[str]:
        parts = []
        for error in errors:
            if self._is_missing(error):
                continue
            text = str(error).strip()
            if text and text not in parts:
                parts.append(text)
        return " | ".join(parts) if parts else None

    def _now_iso(self) -> str:
        return datetime.now().isoformat(timespec="seconds")

    def _default_metrics(self) -> dict[str, Any]:
        return {
            "r_row": 0.0,
            "p_row": 0.0,
            "r_col": 0.0,
            "p_col": 0.0,
            "N_perfect_row": 0,
            "N_perfect_col": 0,
        }

    def _json_safe(self, value: Any) -> Any:
        if value is None:
            return None

        try:
            if pd.isna(value):
                return None
        except Exception:
            pass

        if isinstance(value, (str, int, float, bool)):
            return value

        if hasattr(value, "isoformat"):
            try:
                return value.isoformat()
            except Exception:
                pass

        return str(value)

    def _normalize_result_df(self, result: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if result is None or not isinstance(result, pd.DataFrame):
            return None
        return result.loc[:, ~result.columns.duplicated()].copy()

    def _serialize_result_df(self, result: Optional[pd.DataFrame]) -> Optional[str]:
        result = self._normalize_result_df(result)
        if result is None:
            return None

        payload = {
            "format": "dataframe_split",
            "version": 1,
            "columns": [str(c) for c in result.columns.tolist()],
            "data": [
                [self._json_safe(value) for value in row]
                for row in result.itertuples(index=False, name=None)
            ],
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    def _deserialize_result_df(self, value: Any) -> Optional[pd.DataFrame]:
        if self._is_missing(value):
            return None

        if isinstance(value, pd.DataFrame):
            return self._normalize_result_df(value)

        if not isinstance(value, str):
            return None

        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            return None

        if not isinstance(payload, dict) or payload.get("format") != "dataframe_split":
            return None

        columns = payload.get("columns", [])
        data = payload.get("data", [])

        try:
            df = pd.DataFrame(data=data, columns=columns)
        except Exception:
            logger.warning("Unable to deserialize a stored query result row.")
            return None

        return self._normalize_result_df(df)

    def _read_code_tag(self) -> Optional[str]:
        try:
            with open("tag.txt", "r", encoding="utf-8") as file:
                lines = [line.strip() for line in file if line.strip()]
            for line in reversed(lines):
                if line.startswith("v"):
                    return line[1:]
            return lines[-1] if lines else None
        except Exception as exc:
            logger.warning("Unable to read tag.txt: %s", exc)
            return None

    def _extract_prompt_version(self) -> str:
        prompt_name = os.path.basename(str(self.prompts_path))
        if "prompts_" in prompt_name:
            return prompt_name.split("prompts_", 1)[1].rsplit(".json", 1)[0]
        return os.path.splitext(prompt_name)[0]

    def _resolve_lang_type(self, label: Any, gold: bool) -> str:
        if gold:
            return "sql"
        if self.original_lang_type == "python" and label == "simple":
            return "sql"
        return self.original_lang_type

    def create_conn(self) -> Optional[sa.engine.base.Engine]:
        """Function to create a connection with ALeRCE's SQL database."""
        try:
            params = _get_sql_params()
        except Exception as exc:
            logger.exception("Unable to load SQL connection parameters: %s", exc)
            return None

        if self.t_conn == 2:
            return sa.create_engine(
                f"postgresql+psycopg2://{params['user']}:{params['password']}@"
                f"{params['host']}/{params['dbname']}",
                poolclass=sa.pool.NullPool,
            )

        if self.t_conn == 10:
            return sa.create_engine(
                f"postgresql+psycopg2://{USER_10}:{PASS_10}@"
                f"{params['host']}/{params['dbname']}",
                poolclass=sa.pool.NullPool,
            )

        logger.error("Time not available for SQL connection: %s", self.t_conn)
        return None

    def run_query(
        self,
        specified_format: str,
        formatted_response: str,
        engine: sa.engine.base.Engine,
    ):
        """Run a SQL query or Python-to-SQL query in the database."""
        results = None
        error = None

        if self._is_missing(formatted_response):
            return None, "Empty query received."

        if specified_format == "sql":
            try:
                results = pd.read_sql_query(str(formatted_response), con=engine)
            except Exception as exc:
                error = self._stringify_error(exc)
                logger.exception("Running SQL exception in run_query: %s", exc)

        elif specified_format == "python":
            try:
                execution_scope = {"__builtins__": __builtins__}
                exec(str(formatted_response), execution_scope)
                full_query = execution_scope.get("full_query")
                if self._is_missing(full_query):
                    error = "No 'full_query' variable in Python code."
                    logger.error(error)
                else:
                    results = pd.read_sql_query(str(full_query), con=engine)
            except Exception as exc:
                error = self._stringify_error(exc)
                logger.exception("Running Python exception in run_query: %s", exc)

        else:
            error = "No valid format specified."
            logger.error(error)

        return results, error

    def run_sql_alerce(self, sql: str, label: str, gold: bool):
        """Execute the SQL query at the ALeRCE database and return the result."""
        engine = self.create_conn()
        if engine is None:
            return None, "Database engine could not be created."

        query = None
        error = None
        lang_type = self._resolve_lang_type(label, gold)

        try:
            for attempt in range(1, self.n_tries + 1):
                try:
                    with engine.begin() as conn:
                        query, error = self.run_query(lang_type, sql, conn)
                    if error is None:
                        break
                    logger.warning(
                        "SQL execution failed on attempt %s/%s: %s",
                        attempt,
                        self.n_tries,
                        error,
                    )
                except Exception as exc:
                    error = self._stringify_error(exc)
                    logger.exception(
                        "Unhandled exception on SQL execution attempt %s/%s: %s",
                        attempt,
                        self.n_tries,
                        exc,
                    )
        finally:
            try:
                engine.dispose()
            except Exception as exc:
                logger.warning("Unable to dispose SQL engine cleanly: %s", exc)

        return query, error

    def safe_to_csv(self, df: pd.DataFrame, path: str):
        """Safely write a DataFrame to CSV using a temporary file and rollover backup."""
        dir_name = os.path.dirname(path) or "."
        os.makedirs(dir_name, exist_ok=True)

        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".csv")
        os.close(fd)

        try:
            df.to_csv(tmp_path, index=False)
            if os.path.exists(path):
                shutil.copy2(path, f"{path}.bak")
            os.replace(tmp_path, path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            logger.exception("Failed writing CSV backup at %s", path)
            raise

    def _compute_column_metrics(
        self,
        query_pred: Optional[pd.DataFrame],
        query_gold: Optional[pd.DataFrame],
    ) -> tuple[float, float, int]:
        query_pred = self._normalize_result_df(query_pred)
        query_gold = self._normalize_result_df(query_gold)

        if query_pred is None or query_gold is None:
            return 0.0, 0.0, 0

        cols_pred = query_pred.columns.values.tolist()
        cols_gold = query_gold.columns.values.tolist()

        true_pred_column = sum(1 for col in cols_pred if col in cols_gold)
        true_gold_column = sum(1 for col in cols_gold if col in cols_pred)

        n_cols_pred = len(cols_pred)
        n_cols_gold = len(cols_gold)

        r_col = 0.0 if n_cols_gold == 0 else true_pred_column / n_cols_gold
        p_col = 0.0 if n_cols_pred == 0 else true_gold_column / n_cols_pred
        n_perfect_col = 1 if r_col == 1 else 0
        return float(r_col), float(p_col), int(n_perfect_col)

    def _compute_metrics(
        self,
        query_pred: Optional[pd.DataFrame],
        query_gold: Optional[pd.DataFrame],
    ) -> dict[str, Any]:
        query_pred = self._normalize_result_df(query_pred)
        query_gold = self._normalize_result_df(query_gold)

        if query_pred is None or query_gold is None:
            return self._default_metrics()

        r_col, p_col, n_perfect_col = self._compute_column_metrics(query_pred, query_gold)
        r_row, p_row, n_perfect_row = compute_row_metrics(query_pred, query_gold)
        return {
            "r_row": r_row,
            "p_row": p_row,
            "r_col": r_col,
            "p_col": p_col,
            "N_perfect_row": n_perfect_row,
            "N_perfect_col": n_perfect_col,
        }

    def _build_context_lookup(self, df: pd.DataFrame) -> dict[str, dict[str, Any]]:
        required_columns = {"req_id", "gold_query"}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in dataset DataFrame: {sorted(missing)}")

        working = df.copy()
        if "difficulty" not in working.columns:
            working["difficulty"] = None
        if "request" not in working.columns:
            working["request"] = None

        working["_query_key"] = working["req_id"].apply(self._to_query_key)
        working = working[working["_query_key"].notna()].copy()

        duplicates = working[working["_query_key"].duplicated(keep=False)]["_query_key"].unique()
        if len(duplicates) > 0:
            logger.warning(
                "Duplicate req_id values found in the dataset. Keeping the first occurrence for: %s",
                list(duplicates),
            )
            working = working.drop_duplicates("_query_key", keep="first")

        context_lookup = {}
        for _, row in working.iterrows():
            context_lookup[row["_query_key"]] = {
                "query_id": row["req_id"],
                "gold_query": row["gold_query"],
                "difficulty": row.get("difficulty"),
                "request": row.get("request"),
            }
        return context_lookup

    def _build_prediction_lookup(
        self,
        sql_preds: pd.DataFrame,
        allowed_keys: Optional[set[str]] = None,
    ) -> tuple[dict[tuple[str, int], dict[str, Any]], list[str]]:
        required_columns = {"query_id", "query_run"}
        missing = required_columns - set(sql_preds.columns)
        if missing:
            raise ValueError(
                f"Missing required columns in predictions DataFrame: {sorted(missing)}"
            )

        working = sql_preds.copy()
        for optional_column in [
            "sql_query",
            "tab_schema",
            "label",
            "query_gen_time",
            "query_gen_date",
        ]:
            if optional_column not in working.columns:
                working[optional_column] = None

        working["_query_key"] = working["query_id"].apply(self._to_query_key)
        working["query_run"] = working["query_run"].apply(self._to_run_number)
        working["_source_order"] = range(len(working))

        working = working[working["_query_key"].notna() & working["query_run"].notna()].copy()
        if allowed_keys is not None:
            working = working[working["_query_key"].isin(allowed_keys)].copy()

        duplicates = working.duplicated(subset=["_query_key", "query_run"], keep=False)
        if duplicates.any():
            duplicate_rows = (
                working.loc[duplicates, ["_query_key", "query_run"]]
                .drop_duplicates()
                .values.tolist()
            )
            logger.warning(
                "Duplicate predictions found for query_id/query_run pairs. Keeping the last entry for: %s",
                duplicate_rows,
            )

        working = working.sort_values(["_query_key", "query_run", "_source_order"])
        working = working.drop_duplicates(subset=["_query_key", "query_run"], keep="last")

        query_order = list(dict.fromkeys(working["_query_key"].tolist()))
        pred_lookup = {}
        for _, row in working.iterrows():
            pred_lookup[(row["_query_key"], int(row["query_run"]))] = {
                "query_id": row["query_id"],
                "query_run": int(row["query_run"]),
                "sql_query": row.get("sql_query"),
                "tab_schema": row.get("tab_schema"),
                "label": row.get("label"),
                "query_gen_time": row.get("query_gen_time"),
                "query_gen_date": row.get("query_gen_date"),
            }

        return pred_lookup, query_order

    def _build_state_dataframe(
        self,
        query_order: list[str],
        pred_lookup: dict[tuple[str, int], dict[str, Any]],
        context_lookup: dict[str, dict[str, Any]],
        total_exps: int,
    ) -> pd.DataFrame:
        tag = self._read_code_tag()
        prompt_version = self._extract_prompt_version()
        rows = []

        for query_key in query_order:
            context = context_lookup.get(query_key, {})
            query_id = context.get("query_id", query_key)
            request_text = context.get("request")

            gold_row = {col: None for col in self.METRICS_COLUMNS}
            gold_row.update(
                {
                    "code_tag": tag,
                    "query_id": query_id,
                    "query_run": 0,
                    "sql_query": context.get("gold_query"),
                    "request_text": request_text,
                }
            )
            rows.append(gold_row)

            for exp in range(1, total_exps + 1):
                pred_source = pred_lookup.get((query_key, exp), {})
                pred_row = {col: None for col in self.METRICS_COLUMNS}
                pred_row.update(
                    {
                        "code_tag": tag,
                        "llm_used": self.llm,
                        "prompt_version": prompt_version,
                        "query_id": query_id,
                        "query_run": exp,
                        "sql_query": pred_source.get("sql_query"),
                        "tab_schema": pred_source.get("tab_schema"),
                        "label": pred_source.get("label"),
                        "query_gen_time": pred_source.get("query_gen_time"),
                        "query_gen_date": pred_source.get("query_gen_date"),
                        "request_text": request_text,
                    }
                )
                rows.append(pred_row)

        return pd.DataFrame(rows, columns=self.METRICS_COLUMNS)

    def _ensure_state_columns(self, state: pd.DataFrame) -> pd.DataFrame:
        working = state.copy()
        unnamed_columns = [col for col in working.columns if str(col).startswith("Unnamed:")]
        if unnamed_columns:
            working = working.drop(columns=unnamed_columns)

        for column in self.METRICS_COLUMNS:
            if column not in working.columns:
                working[column] = None

        return working[self.METRICS_COLUMNS]

    def _merge_backup_with_template(
        self,
        backup_df: pd.DataFrame,
        template_df: pd.DataFrame,
    ) -> pd.DataFrame:
        backup_index = self._build_state_index(backup_df)
        used_backup_keys = set()
        merged_rows = []

        for _, template_row in template_df.iterrows():
            key = (
                self._to_query_key(template_row["query_id"]),
                self._to_run_number(template_row["query_run"]),
            )
            merged_row = template_row.to_dict()
            backup_idx = backup_index.get(key)

            if backup_idx is not None:
                used_backup_keys.add(key)
                backup_row = backup_df.loc[backup_idx].to_dict()
                for column in self.METRICS_COLUMNS:
                    if not self._is_missing(backup_row.get(column)):
                        merged_row[column] = backup_row.get(column)

            merged_rows.append(merged_row)

        for _, backup_row in backup_df.iterrows():
            key = (
                self._to_query_key(backup_row["query_id"]),
                self._to_run_number(backup_row["query_run"]),
            )
            if key in used_backup_keys:
                continue

            logger.warning(
                "Keeping backup-only metrics row for query_id=%s run=%s.",
                backup_row.get("query_id"),
                backup_row.get("query_run"),
            )
            merged_rows.append({col: backup_row.get(col) for col in self.METRICS_COLUMNS})

        return pd.DataFrame(merged_rows, columns=self.METRICS_COLUMNS)

    def _build_state_index(self, state: pd.DataFrame) -> dict[tuple[str, int], int]:
        index_lookup = {}
        for idx, row in state.iterrows():
            key = self._to_query_key(row["query_id"])
            run_number = self._to_run_number(row["query_run"])
            if key is None or run_number is None:
                continue
            index_lookup[(key, run_number)] = idx
        return index_lookup

    def _extract_query_order(self, state: pd.DataFrame) -> list[str]:
        query_order = []
        for _, row in state.iterrows():
            if self._to_run_number(row["query_run"]) != 0:
                continue
            key = self._to_query_key(row["query_id"])
            if key is not None and key not in query_order:
                query_order.append(key)

        if query_order:
            return query_order

        for _, row in state.iterrows():
            key = self._to_query_key(row["query_id"])
            if key is not None and key not in query_order:
                query_order.append(key)
        return query_order

    def _extract_run_order_for_query(self, state: pd.DataFrame, query_key: str) -> list[int]:
        run_order = []
        for _, row in state.iterrows():
            if self._to_query_key(row["query_id"]) != query_key:
                continue
            run_number = self._to_run_number(row["query_run"])
            if run_number is None or run_number == 0 or run_number in run_order:
                continue
            run_order.append(run_number)
        return run_order

    def _get_query_context(
        self,
        state: pd.DataFrame,
        state_index: dict[tuple[str, int], int],
        query_key: str,
        context_lookup: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        context = dict(context_lookup.get(query_key, {}))
        gold_index = state_index.get((query_key, 0))

        if gold_index is not None:
            if self._is_missing(context.get("query_id")):
                context["query_id"] = state.at[gold_index, "query_id"]
            if self._is_missing(context.get("gold_query")):
                context["gold_query"] = state.at[gold_index, "sql_query"]
            if self._is_missing(context.get("request")):
                context["request"] = state.at[gold_index, "request_text"]

        return context

    def _row_completed(self, state: pd.DataFrame, index: int) -> bool:
        return not self._is_missing(state.at[index, "sql_date"])

    def _persist_row(
        self,
        state: pd.DataFrame,
        index: int,
        row_values: dict[str, Any],
        bkp_path: str,
    ):
        for column, value in row_values.items():
            if column in state.columns:
                state.at[index, column] = value

        self.new_df = state
        self.safe_to_csv(state, bkp_path)

    def _attempt_self_correction(
        self,
        label: Any,
        sql_query: Any,
        tab_schema: Any,
        request_text: Any,
        error_text: str,
    ) -> tuple[Optional[str], Optional[str]]:
        if self._is_missing(request_text):
            return None, "Self-correction skipped because the original request text is missing."

        lang_type = self._resolve_lang_type(label, gold=False)
        prompt_context = (
            general_context_selfcorr_v1
            if lang_type == "sql"
            else general_context_selfcorr_v1_python
        )

        try:
            corr_prompt = prompt_self_correction_v2(
                gen_task=prompt_context,
                tab_schema=tab_schema,
                req=request_text,
                sql_pred=sql_query,
                error=error_text,
            )
            corrected_response, _ = _api_call(self.llm, self.max_tokens, corr_prompt)
            corrected_query = _format_response(lang_type, corrected_response)

            if self._is_missing(corrected_query):
                return None, "Self-correction returned an empty query."

            return corrected_query, None
        except Exception as exc:
            logger.exception("Self-correction failed: %s", exc)
            return None, self._stringify_error(exc)

    def _execute_gold_row(
        self,
        state: pd.DataFrame,
        index: int,
        context: dict[str, Any],
        bkp_path: str,
    ) -> tuple[Optional[pd.DataFrame], Optional[str]]:
        gold_sql = self._coalesce(state.at[index, "sql_query"], context.get("gold_query"))
        gold_start = time.time()

        if self._is_missing(gold_sql):
            query_gold = None
            error_gold = "Missing gold SQL query for this request."
        else:
            query_gold, error_gold = self.run_sql_alerce(str(gold_sql), context.get("difficulty"), True)

        gold_time = time.time() - gold_start
        gold_date = self._now_iso()
        query_gold = self._normalize_result_df(query_gold)
        error_gold = self._stringify_error(error_gold)

        row_values = {
            "sql_query": gold_sql,
            "tab_schema": None,
            "label": None,
            "query_gen_time": None,
            "query_gen_date": None,
            "request_text": self._coalesce(state.at[index, "request_text"], context.get("request")),
            "query_results": self._serialize_result_df(query_gold),
            "query_error": error_gold,
            "sql_time": gold_time,
            "sql_date": gold_date,
        }

        if query_gold is not None and error_gold is None:
            row_values.update(
                {
                    "r_row": 1.0,
                    "p_row": 1.0,
                    "r_col": 1.0,
                    "p_col": 1.0,
                    "N_perfect_row": 1,
                    "N_perfect_col": 1,
                }
            )
        else:
            row_values.update(self._default_metrics())

        self._persist_row(state, index, row_values, bkp_path)
        return query_gold, error_gold

    def _execute_prediction_row(
        self,
        state: pd.DataFrame,
        index: int,
        context: dict[str, Any],
        pred_source: Optional[dict[str, Any]],
        gold_df: Optional[pd.DataFrame],
        gold_error: Optional[str],
        bkp_path: str,
    ):
        pred_source = pred_source or {}

        base_query = self._coalesce(state.at[index, "sql_query"], pred_source.get("sql_query"))
        label = self._coalesce(state.at[index, "label"], pred_source.get("label"))
        tab_schema = self._coalesce(state.at[index, "tab_schema"], pred_source.get("tab_schema"))
        query_gen_time = self._coalesce(
            state.at[index, "query_gen_time"], pred_source.get("query_gen_time")
        )
        query_gen_date = self._coalesce(
            state.at[index, "query_gen_date"], pred_source.get("query_gen_date")
        )
        request_text = self._coalesce(state.at[index, "request_text"], context.get("request"))

        pred_start = time.time()
        query_pred = None
        error_pred = None
        executed_query = base_query

        if self._is_missing(base_query):
            error_pred = "Missing predicted SQL query for this run."
        elif self._is_missing(label):
            error_pred = "Missing difficulty label for this run."
        else:
            query_pred, error_pred = self.run_sql_alerce(str(base_query), label, False)
            error_pred = self._stringify_error(error_pred)

            if self.self_corr and (error_pred is not None or query_pred is None):
                original_error = error_pred or "Predicted query did not produce a result."
                corrected_query, correction_error = self._attempt_self_correction(
                    label=label,
                    sql_query=base_query,
                    tab_schema=tab_schema,
                    request_text=request_text,
                    error_text=original_error,
                )

                if corrected_query is not None:
                    executed_query = corrected_query
                    query_pred, error_pred = self.run_sql_alerce(corrected_query, label, False)
                    error_pred = self._stringify_error(error_pred)
                    if error_pred is not None:
                        error_pred = self._merge_errors(
                            error_pred,
                            f"Original execution error: {original_error}",
                        )
                elif correction_error is not None:
                    error_pred = self._merge_errors(error_pred, f"Self-correction failed: {correction_error}")

        pred_time = time.time() - pred_start
        pred_date = self._now_iso()

        query_pred = self._normalize_result_df(query_pred)
        metrics = self._default_metrics()

        if gold_df is not None and gold_error is None and query_pred is not None and error_pred is None:
            metrics = self._compute_metrics(query_pred, gold_df)

        final_error = error_pred
        if gold_error is not None:
            final_error = self._merge_errors(
                final_error,
                f"Gold query unavailable for metric comparison: {gold_error}",
            )

        row_values = {
            "sql_query": executed_query,
            "tab_schema": tab_schema,
            "label": label,
            "query_gen_time": query_gen_time,
            "query_gen_date": query_gen_date,
            "request_text": request_text,
            "query_results": self._serialize_result_df(query_pred),
            "query_error": final_error,
            "sql_time": pred_time,
            "sql_date": pred_date,
        }
        row_values.update(metrics)

        self._persist_row(state, index, row_values, bkp_path)

    def run_metrics(
        self,
        sql_preds_path: str,
        df: pd.DataFrame,
        total_exps: int = 10,
        restart: bool = False,
    ):
        """Run the evaluation experiments with resumable and reproducible state."""
        if total_exps <= 0:
            raise ValueError("total_exps must be greater than 0.")

        file_path = (
            f"experiments/metrics_{self.llm}_{datetime.now().isoformat(timespec='seconds')}.csv"
            .replace(":", "-")
        )
        bkp_path = "experiments/bkp_metrics.csv"
        backup_exists = os.path.exists(bkp_path)

        state = None
        final_exception = None

        try:
            context_lookup = {}
            pred_lookup = {}
            query_order = []

            try:
                context_lookup = self._build_context_lookup(df)
            except Exception as exc:
                if restart and backup_exists:
                    logger.warning(
                        "Unable to rebuild dataset context during restart. Falling back to backup-only metadata: %s",
                        exc,
                    )
                else:
                    raise

            try:
                sql_preds = pd.read_csv(sql_preds_path)
                pred_lookup, query_order = self._build_prediction_lookup(
                    sql_preds,
                    allowed_keys=set(context_lookup.keys()) if context_lookup else None,
                )
            except Exception as exc:
                if restart and backup_exists:
                    logger.warning(
                        "Unable to rebuild prediction metadata during restart. Falling back to backup-only metadata: %s",
                        exc,
                    )
                else:
                    raise

            if restart and backup_exists:
                logger.info("Restarting metrics process from backup.")
                backup_state = self._ensure_state_columns(pd.read_csv(bkp_path))

                if query_order:
                    template_state = self._build_state_dataframe(
                        query_order=query_order,
                        pred_lookup=pred_lookup,
                        context_lookup=context_lookup,
                        total_exps=total_exps,
                    )
                    state = self._merge_backup_with_template(backup_state, template_state)
                else:
                    state = backup_state
            else:
                if restart and not backup_exists:
                    logger.warning(
                        "Restart was requested but %s was not found. Starting a fresh metrics run.",
                        bkp_path,
                    )

                if not query_order:
                    raise ValueError(
                        "No predictions were available to build the metrics experiment state."
                    )

                state = self._build_state_dataframe(
                    query_order=query_order,
                    pred_lookup=pred_lookup,
                    context_lookup=context_lookup,
                    total_exps=total_exps,
                )

            self.new_df = state
            self.safe_to_csv(state, bkp_path)

            state_index = self._build_state_index(state)
            query_order = self._extract_query_order(state)

            for query_key in query_order:
                gold_index = state_index.get((query_key, 0))
                if gold_index is None:
                    logger.warning("Skipping query_id=%s because the gold row is missing.", query_key)
                    continue

                context = self._get_query_context(state, state_index, query_key, context_lookup)

                if self._row_completed(state, gold_index):
                    gold_df = self._deserialize_result_df(state.at[gold_index, "query_results"])
                    gold_error = self._stringify_error(state.at[gold_index, "query_error"])
                    if gold_df is None and gold_error is None:
                        logger.warning(
                            "Gold results for query_id=%s could not be deserialized. Re-running the gold query.",
                            query_key,
                        )
                        gold_df, gold_error = self._execute_gold_row(
                            state=state,
                            index=gold_index,
                            context=context,
                            bkp_path=bkp_path,
                        )
                else:
                    gold_df, gold_error = self._execute_gold_row(
                        state=state,
                        index=gold_index,
                        context=context,
                        bkp_path=bkp_path,
                    )

                for exp in self._extract_run_order_for_query(state, query_key):
                    pred_index = state_index.get((query_key, exp))
                    if pred_index is None:
                        logger.warning(
                            "Skipping query_id=%s run=%s because the row is missing in the metrics state.",
                            query_key,
                            exp,
                        )
                        continue

                    if self._row_completed(state, pred_index):
                        continue

                    try:
                        self._execute_prediction_row(
                            state=state,
                            index=pred_index,
                            context=context,
                            pred_source=pred_lookup.get((query_key, exp)),
                            gold_df=gold_df,
                            gold_error=gold_error,
                            bkp_path=bkp_path,
                        )
                    except Exception as exc:
                        logger.exception(
                            "Unhandled error while processing query_id=%s run=%s: %s",
                            query_key,
                            exp,
                            exc,
                        )
                        fallback_source = pred_lookup.get((query_key, exp), {})
                        row_values = {
                            "sql_query": self._coalesce(
                                state.at[pred_index, "sql_query"], fallback_source.get("sql_query")
                            ),
                            "tab_schema": self._coalesce(
                                state.at[pred_index, "tab_schema"], fallback_source.get("tab_schema")
                            ),
                            "label": self._coalesce(
                                state.at[pred_index, "label"], fallback_source.get("label")
                            ),
                            "query_gen_time": self._coalesce(
                                state.at[pred_index, "query_gen_time"],
                                fallback_source.get("query_gen_time"),
                            ),
                            "query_gen_date": self._coalesce(
                                state.at[pred_index, "query_gen_date"],
                                fallback_source.get("query_gen_date"),
                            ),
                            "request_text": self._coalesce(
                                state.at[pred_index, "request_text"], context.get("request")
                            ),
                            "query_results": None,
                            "query_error": self._stringify_error(exc),
                            "sql_time": 0.0,
                            "sql_date": self._now_iso(),
                        }
                        row_values.update(self._default_metrics())
                        self._persist_row(state, pred_index, row_values, bkp_path)

            logger.info("Process ended. Saving final metrics file.")

        except Exception as exc:
            final_exception = exc
            logger.exception("An error has occurred while running metrics: %s", exc)

        finally:
            if state is not None:
                self.new_df = state
                try:
                    self.safe_to_csv(state, bkp_path)
                except Exception:
                    logger.exception("Unable to refresh the metrics backup at shutdown.")

                try:
                    self.safe_to_csv(state, file_path)
                except Exception:
                    logger.exception("Unable to save the final metrics CSV at %s", file_path)

            if final_exception is not None and state is None:
                raise final_exception
