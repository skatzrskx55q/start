import pandas as pd
import requests
import re
import base64
import copy
from io import BytesIO
from sentence_transformers import SentenceTransformer, util
import pymorphy2
import functools
import os
from itertools import product
from bs4 import BeautifulSoup

def get_github_headers():
    token = os.getenv("GITHUB_TOKEN")
    if token:
        return {"Authorization": f"token {token}"}
    return {}

# ---------- Модель и морфологический разбор ----------
@functools.lru_cache(maxsize=1)
def get_model():
    hf_model_id = "sentence-transformers/all-MiniLM-L12-v2"
    return SentenceTransformer(hf_model_id)

@functools.lru_cache(maxsize=1)
def get_morph():
    return pymorphy2.MorphAnalyzer()

def preprocess(text):
    return re.sub(r"\s+", " ", str(text).lower().strip())

def lemmatize(word):
    return get_morph().parse(word)[0].normal_form

@functools.lru_cache(maxsize=10000)
def lemmatize_cached(word):
    return lemmatize(word)

SYNONYM_GROUPS = []  # Пока пусто
SYNONYM_DICT = {}
for group in SYNONYM_GROUPS:
    lemmas = {lemmatize(w.lower()) for w in group}
    for lemma in lemmas:
        SYNONYM_DICT[lemma] = lemmas

DEFAULT_PARSE_PROFILE = {
    "search": {
        "split_newline": True,
        "split_pipe": True,
        "split_slash": True,
    },
    "filter": {
        "split_newline": True,
        "split_pipe": True,
    },
}


def _resolve_parse_profile(parse_profile=None):
    profile = copy.deepcopy(DEFAULT_PARSE_PROFILE)
    if parse_profile is None:
        return profile
    if not isinstance(parse_profile, dict):
        raise TypeError("parse_profile должен быть словарём.")

    allowed_sections = set(profile.keys())
    unknown_sections = set(parse_profile.keys()) - allowed_sections
    if unknown_sections:
        raise KeyError(f"Неизвестные секции parse_profile: {sorted(unknown_sections)}")

    for section, overrides in parse_profile.items():
        if overrides is None:
            continue
        if not isinstance(overrides, dict):
            raise TypeError(f"Секция parse_profile.{section} должна быть словарём.")

        allowed_keys = set(profile[section].keys())
        unknown_keys = set(overrides.keys()) - allowed_keys
        if unknown_keys:
            raise KeyError(
                f"Неизвестные ключи parse_profile.{section}: {sorted(unknown_keys)}"
            )

        for key, value in overrides.items():
            if not isinstance(value, bool):
                raise TypeError(f"Значение parse_profile.{section}.{key} должно быть bool.")
            profile[section][key] = value

    return profile

# ---------- Универсальное разбиение примеров ----------
@functools.lru_cache(maxsize=5000)
def split_by_slash(text: str, split_pipe: bool = True, split_slash: bool = True):
    """Разворачивает варианты по / и сегменты по | внутри одной строки."""
    text = text.strip()
    if not text:
        return []

    if split_pipe:
        segments = [seg.strip() for seg in text.split("|") if seg.strip()]
    else:
        segments = [text]
    all_phrases = []

    for segment in segments:
        segment = re.sub(r"\s+", " ", segment).strip()
        if not segment:
            continue

        if not split_slash:
            all_phrases.append(segment)
            continue

        parts = []
        last_idx = 0

        for match in re.finditer(r"\b[\w-]+(?:/[\w-]+)+\b", segment):
            if match.start() > last_idx:
                prefix = segment[last_idx:match.start()].strip()
                if prefix:
                    parts.append([prefix])

            options = [opt.strip() for opt in match.group(0).split("/") if opt.strip()]
            parts.append(options)
            last_idx = match.end()

        if last_idx < len(segment):
            suffix = segment[last_idx:].strip()
            if suffix:
                parts.append([suffix])

        if not parts:
            all_phrases.append(segment)
            continue

        for combination in product(*parts):
            combined = re.sub(r"\s+", " ", " ".join(combination)).strip()
            if combined:
                all_phrases.append(combined)

    # Preserve order, remove duplicates from slash/pipe expansion.
    return list(dict.fromkeys(all_phrases))


def split_examples(cell, split_newline: bool = True, split_pipe: bool = True, split_slash: bool = True):
    if pd.isna(cell) or not isinstance(cell, str):
        return []

    if split_newline:
        lines = [line.strip() for line in cell.split("\n") if line.strip()]
    else:
        line = cell.strip()
        lines = [line] if line else []

    result = []
    for line in lines:
        result.extend(
            split_by_slash(
                line,
                split_pipe=split_pipe,
                split_slash=split_slash,
            )
        )
    # Preserve order, remove duplicates inside one cell.
    return list(dict.fromkeys([item for item in result if item]))

def _extract_index_from_suffix(col_name: str, prefix: str) -> int:
    suffix = col_name[len(prefix):]
    if suffix.isdigit():
        return int(suffix)
    return 10**9

def _sorted_prefixed_columns(columns, prefix: str):
    pattern = re.compile(rf"^{re.escape(prefix.lower())}\d+$")
    matched = [c for c in columns if pattern.match(c.lower())]
    return sorted(matched, key=lambda c: _extract_index_from_suffix(c.lower(), prefix.lower()))

def _ensure_comment_column(df, target_name: str = "comment1"):
    comment_cols = _sorted_prefixed_columns(df.columns, "comment")
    if comment_cols:
        df[target_name] = df[comment_cols[0]].fillna("").astype(str)
    else:
        df[target_name] = ""
    return df

def _get_unified_column_lists(df):
    search_cols = _sorted_prefixed_columns(df.columns, "search")
    filter_cols = _sorted_prefixed_columns(df.columns, "display_filter")
    display_cols = _sorted_prefixed_columns(df.columns, "display")
    comment_cols = _sorted_prefixed_columns(df.columns, "comment")
    return search_cols, filter_cols, display_cols, comment_cols

def _set_unified_attrs(df):
    search_cols, filter_cols, display_cols, comment_cols = _get_unified_column_lists(df)
    df.attrs["search_cols"] = search_cols
    df.attrs["filter_cols"] = filter_cols
    df.attrs["display_cols"] = display_cols
    df.attrs["comment_cols"] = comment_cols
    return df

def _resolve_result_columns(df, filter_cols=None, display_cols=None, comment_col=None):
    default_filter_cols = df.attrs.get("filter_cols") or _sorted_prefixed_columns(df.columns, "display_filter")
    default_display_cols = df.attrs.get("display_cols") or _sorted_prefixed_columns(df.columns, "display")
    default_comment_cols = df.attrs.get("comment_cols") or _sorted_prefixed_columns(df.columns, "comment")

    use_filter_cols = list(filter_cols) if filter_cols is not None else list(default_filter_cols)
    use_display_cols = list(display_cols) if display_cols is not None else list(default_display_cols)

    use_comment_col = comment_col
    if use_comment_col is None:
        use_comment_col = default_comment_cols[0] if default_comment_cols else "comment1"

    return use_filter_cols, use_display_cols, use_comment_col


def _value_to_text(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)

def _split_filter_values(value: str, split_newline: bool = True, split_pipe: bool = True):
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    # Support configurable separators for filter values.
    parts = []
    chunks = text.split("\n") if split_newline else [text]
    for chunk in chunks:
        candidate_parts = chunk.split("|") if split_pipe else [chunk]
        for part in candidate_parts:
            part = part.strip()
            if part:
                parts.append(part)
    return parts

def _explode_search_rows(
    df,
    search_cols,
    intermediate_col="examples_split",
    split_newline: bool = True,
    split_pipe: bool = True,
    split_slash: bool = True,
):
    phrase_parts_col = []
    for _, row in df.iterrows():
        row_parts = []
        for col in search_cols:
            row_parts.extend(
                split_examples(
                    row[col],
                    split_newline=split_newline,
                    split_pipe=split_pipe,
                    split_slash=split_slash,
                )
            )
        row_parts = [p for p in row_parts if p]
        if not row_parts:
            row_parts = [""]
        # Remove duplicates per source row (e.g. same phrase in search1 and search2).
        phrase_parts_col.append(list(dict.fromkeys(row_parts)))

    df[intermediate_col] = phrase_parts_col
    df = df.explode(intermediate_col)
    df = df[df[intermediate_col].notna()]
    df = df.rename(columns={intermediate_col: "phrase"})
    df["phrase"] = df["phrase"].astype(str).str.strip()
    return df[df["phrase"] != ""]

def _structured_search_results(
    query,
    df,
    threshold=0.5,
    top_k=None,
    filter_cols=None,
    display_cols=None,
    comment_col=None,
    include_semantic=True,
    include_keyword=True,
):
    use_filter_cols, use_display_cols, use_comment_col = _resolve_result_columns(
        df, filter_cols=filter_cols, display_cols=display_cols, comment_col=comment_col
    )

    query_proc = preprocess(query)
    query_words = re.findall(r"\w+", query_proc)
    query_lemmas = [lemmatize_cached(w) for w in query_words]
    sims = None
    if include_semantic:
        model = get_model()
        query_emb = model.encode(query_proc, convert_to_tensor=True)
        phrase_embs = df.attrs["phrase_embs"]
        sims = util.pytorch_cos_sim(query_emb, phrase_embs)[0]

    semantic_results = []
    keyword_results = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        if include_semantic and sims is not None:
            score_float = float(sims[idx])
            if score_float >= threshold:
                semantic_results.append(
                    {
                        "score": score_float,
                        "phrase": row["phrase"],
                        "filters": {col: _value_to_text(row.get(col, "")) for col in use_filter_cols},
                        "displays": {col: _value_to_text(row.get(col, "")) for col in use_display_cols},
                        "comment": _value_to_text(row.get(use_comment_col, "")),
                        "original_index": row["original_index"],
                    }
                )

        if include_keyword:
            lemma_match = all(
                any(ql in SYNONYM_DICT.get(pl, {pl}) for pl in row.phrase_lemmas)
                for ql in query_lemmas
            )
            partial_match = all(q in row.phrase_proc for q in query_words)
            if lemma_match or partial_match:
                keyword_results.append(
                    {
                        "phrase": row["phrase"],
                        "filters": {col: _value_to_text(row.get(col, "")) for col in use_filter_cols},
                        "displays": {col: _value_to_text(row.get(col, "")) for col in use_display_cols},
                        "comment": _value_to_text(row.get(use_comment_col, "")),
                        "original_index": row["original_index"],
                    }
                )

    semantic_results = sorted(semantic_results, key=lambda x: x["score"], reverse=True)
    if top_k is not None:
        semantic_results = semantic_results[:top_k]
    return semantic_results, keyword_results


def _normalized_columns(columns):
    ordered = []
    seen = set()
    for col in columns or []:
        if col is None:
            continue
        name = str(col).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return ordered


def prepare_runtime_dataframe(
    df_input,
    search_cols,
    filter_cols=None,
    display_cols=None,
    comment_col=None,
    parse_profile=None,
):
    """
    Подготавливает runtime-датасет для поиска/фильтров из таблицы произвольной структуры.
    Пользовательские колонки задаются через mapping.
    """
    if df_input is None or df_input.empty:
        raise ValueError("Таблица пуста.")

    search_cols = _normalized_columns(search_cols)
    filter_cols = _normalized_columns(filter_cols)
    display_cols = _normalized_columns(display_cols)
    comment_col = str(comment_col).strip() if comment_col is not None else None
    if comment_col == "":
        comment_col = None

    if not search_cols:
        raise ValueError("Нужно выбрать хотя бы одну поисковую колонку.")

    missing = [c for c in search_cols + filter_cols + display_cols if c not in df_input.columns]
    if comment_col and comment_col not in df_input.columns:
        missing.append(comment_col)
    if missing:
        raise KeyError(f"Не найдены колонки в таблице: {missing}")

    profile = _resolve_parse_profile(parse_profile)
    search_profile = profile["search"]
    filter_profile = profile["filter"]

    df = df_input.copy()
    for col in search_cols + filter_cols + display_cols:
        df[col] = df[col].fillna("").astype(str)

    if comment_col:
        df["comment1"] = df[comment_col].fillna("").astype(str)
        attrs_comment_cols = [comment_col]
    else:
        df["comment1"] = ""
        attrs_comment_cols = ["comment1"]

    df = df.reset_index(drop=True)
    df["original_index"] = [str(i) for i in df.index]

    df = _explode_search_rows(
        df,
        search_cols,
        intermediate_col="phrase_list",
        split_newline=search_profile["split_newline"],
        split_pipe=search_profile["split_pipe"],
        split_slash=search_profile["split_slash"],
    ).reset_index(drop=True)

    df["phrase_proc"] = df["phrase"].apply(preprocess)
    df["phrase_lemmas"] = df["phrase_proc"].apply(
        lambda t: {lemmatize_cached(w) for w in re.findall(r"\w+", t)}
    )

    if display_cols:
        df["phrase_full"] = df[display_cols[0]].astype(str)
    else:
        df["phrase_full"] = df["phrase"]

    if filter_cols:
        def _collect_topics(row):
            values = []
            for col in filter_cols:
                values.extend(
                    _split_filter_values(
                        row[col],
                        split_newline=filter_profile["split_newline"],
                        split_pipe=filter_profile["split_pipe"],
                    )
                )
            return list(dict.fromkeys(values))

        df["topics"] = df.apply(_collect_topics, axis=1)
    else:
        df["topics"] = [[] for _ in range(len(df))]

    original_examples_map = {}
    for _, row in df.iterrows():
        orig_idx = row["original_index"]
        original_examples_map.setdefault(orig_idx, set()).add(row["phrase"])

    df.attrs["search_cols"] = list(search_cols)
    df.attrs["filter_cols"] = list(filter_cols)
    df.attrs["display_cols"] = list(display_cols)
    df.attrs["comment_cols"] = attrs_comment_cols
    df.attrs["parse_profile"] = profile
    df.attrs["original_examples_map"] = original_examples_map

    model = get_model()
    df.attrs["phrase_embs"] = model.encode(df["phrase_proc"].tolist(), convert_to_tensor=True)

    return df

# ---------- Унифицированная загрузка табличных данных ----------

def load_unified_excel(url, source_id="0", parse_profile=None):
    profile = _resolve_parse_profile(parse_profile)
    search_profile = profile["search"]
    filter_profile = profile["filter"]

    resp = requests.get(url, headers=get_github_headers())
    if resp.status_code != 200:
        raise ValueError(f"Ошибка загрузки {url}")
    df = pd.read_excel(BytesIO(resp.content))
    search_cols = _sorted_prefixed_columns(df.columns, "search")
    if not search_cols:
        raise KeyError("Не найдены колонки search* для унифицированного формата.")
    filter_cols = _sorted_prefixed_columns(df.columns, "display_filter")
    display_cols = _sorted_prefixed_columns(df.columns, "display")
    for col in search_cols + filter_cols + display_cols:
        df[col] = df[col].fillna("").astype(str)
    df = _ensure_comment_column(df, "comment1")
    if "display1" not in df.columns and display_cols:
        df["display1"] = df[display_cols[0]]
    if "display_filter1" not in df.columns and filter_cols:
        df["display_filter1"] = df[filter_cols[0]]

    df = df.reset_index(drop=True)
    df["original_index"] = [f"{source_id}:{idx}" for idx in df.index]

    df = _explode_search_rows(
        df,
        search_cols,
        intermediate_col="phrase_list",
        split_newline=search_profile["split_newline"],
        split_pipe=search_profile["split_pipe"],
        split_slash=search_profile["split_slash"],
    ).reset_index(drop=True)

    df["phrase_proc"] = df["phrase"].apply(preprocess)
    df["phrase_lemmas"] = df["phrase_proc"].apply(
        lambda t: {lemmatize_cached(w) for w in re.findall(r"\w+", t)}
    )
    df["search1"] = df["phrase"]
    df["search1_proc"] = df["phrase_proc"]
    df["phrase_full"] = df["display1"] if "display1" in df.columns else df["search1"]

    if filter_cols:
        def _collect_topics(row):
            values = []
            for col in filter_cols:
                values.extend(
                    _split_filter_values(
                        row[col],
                        split_newline=filter_profile["split_newline"],
                        split_pipe=filter_profile["split_pipe"],
                    )
                )
            return list(dict.fromkeys(values))
        df["topics"] = df.apply(_collect_topics, axis=1)
    elif "display_filter1" in df.columns:
        df["topics"] = df["display_filter1"].apply(
            lambda v: _split_filter_values(
                v,
                split_newline=filter_profile["split_newline"],
                split_pipe=filter_profile["split_pipe"],
            )
        )
    else:
        df["topics"] = [[] for _ in range(len(df))]

    df.attrs["parse_profile"] = profile
    return df


def load_unified_excels(urls, parse_profile=None):
    profile = _resolve_parse_profile(parse_profile)
    dfs = []
    for i, url in enumerate(urls):
        try:
            dfs.append(load_unified_excel(url, source_id=str(i), parse_profile=profile))
        except Exception as e:
            print(f"Ошибка с {url}: {e}")
            raise

    if not dfs:
        raise ValueError("Не удалось загрузить ни одного файла")

    df_all = pd.concat(dfs, ignore_index=True)
    df_all = _set_unified_attrs(df_all)

    # Строим карту исходных примеров по original_index.
    original_examples_map = {}
    for idx, row in df_all.iterrows():
        orig_idx = row["original_index"]
        if orig_idx not in original_examples_map:
            original_examples_map[orig_idx] = set()
        original_examples_map[orig_idx].add(row["phrase"])

    df_all.attrs["original_examples_map"] = original_examples_map
    df_all.attrs["parse_profile"] = profile

    # Эмбеддинги.
    model = get_model()
    df_all.attrs["phrase_embs"] = model.encode(df_all["phrase_proc"].tolist(), convert_to_tensor=True)

    return df_all

# ---------- Универсальные операции поиска ----------
def _deduplicate_structured_results(results, keep_max_score=False):
    best = {}
    for item in results:
        display_values = item.get("displays", {})
        key = display_values.get("display1") or item.get("phrase", "")
        if not key:
            continue
        if key not in best:
            best[key] = item
            continue
        if keep_max_score:
            old_score = float(best[key].get("score", -1.0))
            new_score = float(item.get("score", -1.0))
            if new_score > old_score:
                best[key] = item
    return list(best.values())


def semantic_search_rows(query, df, threshold=0.5, top_k=None, filter_cols=None, display_cols=None, comment_col=None, deduplicate=False):
    semantic_results, _ = _structured_search_results(
        query,
        df,
        threshold=threshold,
        top_k=top_k,
        filter_cols=filter_cols,
        display_cols=display_cols,
        comment_col=comment_col,
        include_semantic=True,
        include_keyword=False,
    )
    if deduplicate:
        semantic_results = _deduplicate_structured_results(semantic_results, keep_max_score=True)
        semantic_results = sorted(semantic_results, key=lambda x: x["score"], reverse=True)
        if top_k is not None:
            semantic_results = semantic_results[:top_k]
    return semantic_results


def keyword_search_rows(query, df, filter_cols=None, display_cols=None, comment_col=None, deduplicate=False):
    _, keyword_results = _structured_search_results(
        query,
        df,
        filter_cols=filter_cols,
        display_cols=display_cols,
        comment_col=comment_col,
        include_semantic=False,
        include_keyword=True,
    )
    if deduplicate:
        keyword_results = _deduplicate_structured_results(keyword_results, keep_max_score=False)
    return keyword_results


def group_search_results(search_results, df_attrs, search_type="semantic", group_by_filter_cols=None):
    groups = {}
    original_map = df_attrs.get("original_examples_map", {})

    for res in search_results:
        score = res.get("score") if search_type == "semantic" else None
        phrase = res["phrase"]
        filters = res.get("filters", {})
        displays = res.get("displays", {})
        comment = res.get("comment", "")
        orig_idx = res["original_index"]

        key_cols = list(group_by_filter_cols) if group_by_filter_cols is not None else list(filters.keys())
        key = tuple(filters.get(col, "") for col in key_cols)

        if key not in groups:
            groups[key] = {
                "filters": {col: filters.get(col, "") for col in key_cols},
                "displays": displays.copy(),
                "comment": comment,
                "max_score": score,
                "best_phrase": phrase if search_type == "semantic" else None,
                "matched_phrases": {phrase},
                "original_index": orig_idx,
            }
        else:
            groups[key]["matched_phrases"].add(phrase)
            if search_type == "semantic" and score is not None:
                if groups[key]["max_score"] is None or score > groups[key]["max_score"]:
                    groups[key]["max_score"] = score
                    groups[key]["best_phrase"] = phrase
                    groups[key]["original_index"] = orig_idx
                    groups[key]["displays"] = displays.copy()
                    groups[key]["comment"] = comment

    for group in groups.values():
        orig_idx = group["original_index"]
        group["all_phrases"] = sorted(original_map.get(orig_idx, set()))
        del group["original_index"]

    result_list = list(groups.values())
    if search_type == "semantic":
        result_list.sort(
            key=lambda x: x["max_score"] if x["max_score"] is not None else 0,
            reverse=True,
        )
    return result_list


# ========== Работа с текстовыми источниками (Confluence) ==========
def extract_text_from_html(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    raw = soup.get_text(separator="\n")
    lines = [line.strip() for line in raw.split("\n") if line.strip()]
    text = "\n\n".join(lines)
    return text

def chunk_text(text, max_chars=1000, overlap=200):
    paragraphs = text.split("\n")
    chunks = []
    current_chunk = ""
    for p in paragraphs:
        if len(current_chunk) + len(p) < max_chars:
            current_chunk += p + "\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = p + "\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def load_document_data(urls):
    url = urls[0]
    m = re.search(r'/pages/(\d+)', url)
    if not m:
        raise ValueError("Не удалось определить page_id из URL Confluence")
    page_id = m.group(1)

    base = url.split("/wiki")[0]
    api_url = f"{base}/wiki/rest/api/content/{page_id}?expand=body.storage"

    email = os.getenv("CONFLUENCE_EMAIL")
    token = os.getenv("CONFLUENCE_API_TOKEN")
    if not email or not token:
        raise ValueError("Не указаны CONFLUENCE_EMAIL или CONFLUENCE_API_TOKEN")

    auth_pair = f"{email}:{token}"
    b64 = base64.b64encode(auth_pair.encode()).decode()

    headers = {
        "Authorization": f"Basic {b64}",
        "Accept": "application/json"
    }

    resp = requests.get(api_url, headers=headers)
    if resp.status_code != 200:
        raise ValueError(f"Ошибка API Confluence: {resp.status_code}")

    data = resp.json()
    raw_html = data["body"]["storage"]["value"]

    text = extract_text_from_html(raw_html)
    chunks = chunk_text(text)

    if not chunks:
        raise ValueError("Страница Confluence загружена, но текст не извлечён")

    df = pd.DataFrame({"chunk": chunks})
    df["chunk_proc"] = df["chunk"].apply(preprocess)
    df["search1"] = df["chunk"]
    df["display1"] = df["chunk"]
    df["comment1"] = ""
    df = _set_unified_attrs(df)

    model = get_model()
    df.attrs["chunk_embs"] = model.encode(
        df["chunk_proc"].tolist(),
        convert_to_tensor=True
    )

    return df

def semantic_search_document(query, df, top_k=5, threshold=0.3):
    model = get_model()
    query_proc = preprocess(query)
    query_emb = model.encode(query_proc, convert_to_tensor=True)
    chunk_embs = df.attrs["chunk_embs"]
    sims = util.pytorch_cos_sim(query_emb, chunk_embs)[0]
    results = []
    for idx, score in enumerate(sims):
        if float(score) >= threshold:
            results.append((float(score), df.iloc[idx]["chunk"]))
    results = sorted(results, key=lambda x: x[0], reverse=True)[:top_k]
    return results

