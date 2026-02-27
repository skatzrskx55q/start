import hashlib
import hmac
import io
import json
import os
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from utils import (
    DEFAULT_PARSE_PROFILE,
    keyword_search_rows,
    prepare_runtime_dataframe,
    semantic_search_rows,
)


PROJECTS_DIR = Path(".projects")
PROJECT_DATA_DIR = PROJECTS_DIR / "data"


def check_password():
    expected = os.getenv("APP_PASSWORD")
    if not expected:
        st.error("APP_PASSWORD не задан в окружении.")
        return False

    if st.session_state.get("password_correct", False):
        return True

    def password_entered():
        entered = st.session_state.get("password", "")
        ok = hmac.compare_digest(entered, expected)
        st.session_state["password_correct"] = ok
        if ok:
            st.session_state.pop("password", None)

    st.text_input("Пароль", type="password", key="password", on_change=password_entered)
    st.info("После ввода верного пароля первый запуск может занять некоторое время. Пожалуйста, подождите.")
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("Неверный пароль")
    return False


def _slugify(value):
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")
    return slug or "project"


def _file_hash(file_bytes):
    return hashlib.sha256(file_bytes).hexdigest()


def _is_excel(file_name):
    return Path(file_name).suffix.lower() in {".xlsx", ".xlsm", ".xls"}


@st.cache_data(show_spinner=False)
def _get_excel_sheets(file_bytes):
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    return xls.sheet_names


@st.cache_data(show_spinner=False)
def _read_table(file_bytes, file_name, sheet_name=None):
    ext = Path(file_name).suffix.lower()
    if ext in {".xlsx", ".xlsm", ".xls"}:
        target_sheet = sheet_name if sheet_name is not None else 0
        return pd.read_excel(io.BytesIO(file_bytes), sheet_name=target_sheet)

    if ext in {".csv", ".txt"}:
        try:
            return pd.read_csv(io.BytesIO(file_bytes), sep=None, engine="python")
        except UnicodeDecodeError:
            return pd.read_csv(io.BytesIO(file_bytes), sep=None, engine="python", encoding="cp1251")

    raise ValueError(f"Неподдерживаемый формат файла: {ext}")


def _auto_prefixed_columns(columns, prefix):
    pattern = re.compile(rf"^{re.escape(prefix.lower())}\d+$")
    matched = [c for c in columns if pattern.match(str(c).lower())]
    return sorted(matched, key=lambda name: int(re.findall(r"\d+$", name)[0]) if re.findall(r"\d+$", name) else 10**9)


def _auto_mapping(columns):
    search_cols = _auto_prefixed_columns(columns, "search")
    filter_cols = _auto_prefixed_columns(columns, "display_filter")
    display_cols = _auto_prefixed_columns(columns, "display")
    comment_cols = _auto_prefixed_columns(columns, "comment")

    if not search_cols and columns:
        search_cols = [columns[0]]
    if not display_cols and columns:
        display_cols = [columns[0]]

    return {
        "search_cols": search_cols,
        "filter_cols": filter_cols,
        "display_cols": display_cols,
        "comment_col": comment_cols[0] if comment_cols else None,
    }


def _merge_parse_profile(overrides):
    base = json.loads(json.dumps(DEFAULT_PARSE_PROFILE))
    if not overrides:
        return base
    for section, values in overrides.items():
        if section not in base or not isinstance(values, dict):
            continue
        for key, value in values.items():
            if key in base[section]:
                base[section][key] = bool(value)
    return base


def _split_filter_values(value, split_newline=True, split_pipe=True):
    text = str(value).strip()
    if not text:
        return []
    chunks = text.split("\n") if split_newline else [text]
    result = []
    for chunk in chunks:
        parts = chunk.split("|") if split_pipe else [chunk]
        for part in parts:
            part = part.strip()
            if part:
                result.append(part)
    return result


def _result_matches_filters(result, selected_filters, filter_profile):
    for col, selected_values in selected_filters.items():
        if not selected_values:
            continue
        raw_value = result.get("filters", {}).get(col, "")
        values = _split_filter_values(
            raw_value,
            split_newline=filter_profile.get("split_newline", True),
            split_pipe=filter_profile.get("split_pipe", True),
        )
        if not set(values) & set(selected_values):
            return False
    return True


def _collect_filter_options(df_runtime, col, filter_profile):
    options = set()
    if col not in df_runtime.columns:
        return []
    for value in df_runtime[col].tolist():
        for item in _split_filter_values(
            value,
            split_newline=filter_profile.get("split_newline", True),
            split_pipe=filter_profile.get("split_pipe", True),
        ):
            options.add(item)
    return sorted(options)


def _build_config(
    project_name,
    mapping,
    parse_profile,
    search_cfg,
    ui_cfg,
):
    return {
        "project_name": project_name,
        "mapping": mapping,
        "parse_profile": parse_profile,
        "search": search_cfg,
        "ui": ui_cfg,
    }


def _project_registry_path(project_slug):
    return PROJECTS_DIR / f"{project_slug}.json"


def _save_project(project_name, file_name, file_bytes, sheet_name, config):
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    PROJECT_DATA_DIR.mkdir(parents=True, exist_ok=True)

    project_slug = _slugify(project_name)
    file_slug = _slugify(Path(file_name).stem)
    ext = Path(file_name).suffix.lower() or ".bin"
    data_file_name = f"{project_slug}__{file_slug}{ext}"
    data_path = PROJECT_DATA_DIR / data_file_name
    data_path.write_bytes(file_bytes)

    payload = {
        "project_name": project_name,
        "source": {
            "file_name": file_name,
            "sheet_name": sheet_name,
            "data_file": str(data_path.as_posix()),
        },
        "config": config,
    }
    _project_registry_path(project_slug).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _list_saved_projects():
    if not PROJECTS_DIR.exists():
        return []
    return sorted(PROJECTS_DIR.glob("*.json"))


def _load_saved_project(project_manifest_path):
    payload = json.loads(project_manifest_path.read_text(encoding="utf-8"))
    source = payload.get("source", {})
    data_file = source.get("data_file")
    if not data_file:
        raise ValueError("В проекте не указан путь к файлу данных.")
    data_path = Path(data_file)
    if not data_path.exists():
        raise FileNotFoundError(f"Файл данных проекта не найден: {data_path}")
    file_bytes = data_path.read_bytes()
    return payload, file_bytes


@st.cache_resource(show_spinner=False)
def _build_runtime_df(file_hash, file_bytes, file_name, sheet_name, config_json):
    _ = file_hash
    config = json.loads(config_json)
    mapping = config["mapping"]
    parse_profile = config["parse_profile"]

    df_source = _read_table(file_bytes, file_name, sheet_name=sheet_name)
    return prepare_runtime_dataframe(
        df_source,
        search_cols=mapping["search_cols"],
        filter_cols=mapping["filter_cols"],
        display_cols=mapping["display_cols"],
        comment_col=mapping["comment_col"],
        parse_profile=parse_profile,
    )


def _render_result_card(item, idx, title_column, show_score, show_comment, show_phrase, display_cols, filter_cols):
    displays = item.get("displays", {})
    filters = item.get("filters", {})
    phrase = item.get("phrase", "")
    comment = item.get("comment", "")

    if title_column == "phrase":
        title_text = phrase
    else:
        title_text = displays.get(title_column, "") or phrase

    with st.container():
        st.markdown(f"**{idx}. {title_text}**")
        if show_score and "score" in item:
            st.caption(f"Релевантность: {float(item['score']):.2f}")

        if show_phrase:
            st.markdown(f"`match` : {phrase}")

        for col in display_cols:
            value = str(displays.get(col, "")).strip()
            if not value:
                continue
            if col == title_column:
                continue
            st.markdown(f"`{col}` : {value}")

        tags = []
        for col in filter_cols:
            value = str(filters.get(col, "")).strip()
            if value:
                tags.append(f"{col}: {value}")
        if tags:
            st.caption(" | ".join(tags))

        if show_comment and str(comment).strip():
            with st.expander("Комментарий"):
                st.write(comment)


def _render_preview(config, file_name, file_bytes, sheet_name):
    file_hash = _file_hash(file_bytes)
    config_json = json.dumps(config, ensure_ascii=False, sort_keys=True)

    with st.spinner("Подготовка данных для предпросмотра..."):
        df_runtime = _build_runtime_df(
            file_hash=file_hash,
            file_bytes=file_bytes,
            file_name=file_name,
            sheet_name=sheet_name,
            config_json=config_json,
        )

    mapping = config["mapping"]
    parse_profile = config["parse_profile"]
    search_cfg = config["search"]
    ui_cfg = config["ui"]

    st.caption(
        f"Строк после развертки search: {len(df_runtime)} | "
        f"search: {len(mapping['search_cols'])}, "
        f"filter: {len(mapping['filter_cols'])}, "
        f"display: {len(mapping['display_cols'])}"
    )

    selected_filters = {}
    if ui_cfg.get("show_filters", True) and mapping["filter_cols"]:
        with st.expander("Фильтры", expanded=True):
            for col in mapping["filter_cols"]:
                options = _collect_filter_options(df_runtime, col, parse_profile["filter"])
                key = f"{file_hash}_filter_{col}"
                selected_filters[col] = st.multiselect(col, options=options, key=key)

    query_key = f"{file_hash}_query_input"
    last_query_key = f"{file_hash}_last_query"

    with st.form(key=f"{file_hash}_search_form"):
        query_input = st.text_input("Введите запрос", key=query_key)
        submitted = st.form_submit_button("Найти", use_container_width=True)

    if submitted:
        st.session_state[last_query_key] = query_input.strip()

    query = st.session_state.get(last_query_key, "").strip()
    if not query:
        st.info("Введите запрос и нажмите «Найти».")
        return

    semantic_results = semantic_search_rows(
        query,
        df_runtime,
        threshold=float(search_cfg["semantic_threshold"]),
        top_k=int(search_cfg["semantic_top_k"]),
        filter_cols=mapping["filter_cols"],
        display_cols=mapping["display_cols"],
        comment_col="comment1",
        deduplicate=bool(search_cfg["semantic_deduplicate"]),
    )
    semantic_results = [
        item
        for item in semantic_results
        if _result_matches_filters(item, selected_filters, parse_profile["filter"])
    ]

    st.markdown("### Семантический поиск")
    if semantic_results:
        for idx, item in enumerate(semantic_results, start=1):
            _render_result_card(
                item=item,
                idx=idx,
                title_column=ui_cfg["title_column"],
                show_score=True,
                show_comment=ui_cfg["show_comment"],
                show_phrase=ui_cfg["show_matched_phrase"],
                display_cols=mapping["display_cols"],
                filter_cols=mapping["filter_cols"],
            )
    else:
        st.warning("Совпадений не найдено.")

    if search_cfg.get("enable_keyword", True):
        keyword_results = keyword_search_rows(
            query,
            df_runtime,
            filter_cols=mapping["filter_cols"],
            display_cols=mapping["display_cols"],
            comment_col="comment1",
            deduplicate=bool(search_cfg["keyword_deduplicate"]),
        )
        keyword_results = [
            item
            for item in keyword_results
            if _result_matches_filters(item, selected_filters, parse_profile["filter"])
        ]

        st.markdown("### Точный поиск")
        if keyword_results:
            for idx, item in enumerate(keyword_results, start=1):
                _render_result_card(
                    item=item,
                    idx=idx,
                    title_column=ui_cfg["title_column"],
                    show_score=False,
                    show_comment=ui_cfg["show_comment"],
                    show_phrase=ui_cfg["show_matched_phrase"],
                    display_cols=mapping["display_cols"],
                    filter_cols=mapping["filter_cols"],
                )
        else:
            st.info("Совпадений в точном поиске нет.")


def _render_builder():
    st.markdown("### 1) Загрузка таблицы")
    uploaded_file = st.file_uploader("Файл данных", type=["xlsx", "xls", "xlsm", "csv", "txt"])

    file_name = st.session_state.get("builder_file_name")
    file_bytes = st.session_state.get("builder_file_bytes")
    sheet_name = st.session_state.get("builder_sheet_name")

    if uploaded_file is not None:
        file_name = uploaded_file.name
        file_bytes = uploaded_file.getvalue()
        st.session_state["builder_file_name"] = file_name
        st.session_state["builder_file_bytes"] = file_bytes
        if not _is_excel(file_name):
            st.session_state["builder_sheet_name"] = None
            sheet_name = None

    if not file_name or not file_bytes:
        st.info("Загрузите таблицу, чтобы открыть визуальный конструктор.")
        return

    if _is_excel(file_name):
        sheets = _get_excel_sheets(file_bytes)
        default_sheet = sheet_name if sheet_name in sheets else sheets[0]
        sheet_name = st.selectbox("Лист Excel", options=sheets, index=sheets.index(default_sheet))
        st.session_state["builder_sheet_name"] = sheet_name
    else:
        sheet_name = None

    try:
        df_source = _read_table(file_bytes, file_name, sheet_name=sheet_name)
    except Exception as exc:
        st.error(f"Ошибка чтения таблицы: {exc}")
        return

    st.caption(f"Источник: `{file_name}` | Строк: {len(df_source)} | Колонок: {len(df_source.columns)}")
    st.dataframe(df_source.head(20), use_container_width=True)

    columns = [str(col) for col in df_source.columns]
    active_cfg = st.session_state.get("builder_config", {})
    active_mapping = active_cfg.get("mapping", {})
    active_parse = active_cfg.get("parse_profile", {})
    active_search = active_cfg.get("search", {})
    active_ui = active_cfg.get("ui", {})

    auto = _auto_mapping(columns)
    file_hash = _file_hash(file_bytes)

    st.markdown("### 2) Назначение типов колонок")
    search_cols = st.multiselect(
        "Колонки поиска (обязательно)",
        options=columns,
        default=[c for c in active_mapping.get("search_cols", auto["search_cols"]) if c in columns],
        key=f"{file_hash}_search_cols",
    )
    filter_cols = st.multiselect(
        "Колонки фильтров (опционально)",
        options=columns,
        default=[c for c in active_mapping.get("filter_cols", auto["filter_cols"]) if c in columns],
        key=f"{file_hash}_filter_cols",
    )
    display_cols = st.multiselect(
        "Колонки вывода (опционально)",
        options=columns,
        default=[c for c in active_mapping.get("display_cols", auto["display_cols"]) if c in columns],
        key=f"{file_hash}_display_cols",
    )
    comment_options = ["<нет>"] + columns
    comment_default = active_mapping.get("comment_col", auto["comment_col"])
    if comment_default not in comment_options:
        comment_default = "<нет>"
    comment_col = st.selectbox(
        "Колонка комментария",
        options=comment_options,
        index=comment_options.index(comment_default),
        key=f"{file_hash}_comment_col",
    )
    comment_col = None if comment_col == "<нет>" else comment_col

    st.markdown("### 3) Параметры разбиения")
    merged_parse_profile = _merge_parse_profile(active_parse)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Поисковые колонки**")
        search_split_newline = st.checkbox(
            "split `\\n`",
            value=merged_parse_profile["search"]["split_newline"],
            key=f"{file_hash}_search_split_newline",
        )
        search_split_pipe = st.checkbox(
            "split `|`",
            value=merged_parse_profile["search"]["split_pipe"],
            key=f"{file_hash}_search_split_pipe",
        )
        search_split_slash = st.checkbox(
            "split `/`",
            value=merged_parse_profile["search"]["split_slash"],
            key=f"{file_hash}_search_split_slash",
        )
    with col_b:
        st.markdown("**Фильтровые колонки**")
        filter_split_newline = st.checkbox(
            "split `\\n` (filter)",
            value=merged_parse_profile["filter"]["split_newline"],
            key=f"{file_hash}_filter_split_newline",
        )
        filter_split_pipe = st.checkbox(
            "split `|` (filter)",
            value=merged_parse_profile["filter"]["split_pipe"],
            key=f"{file_hash}_filter_split_pipe",
        )

    st.markdown("### 4) Настройки поиска и UI")
    col_c, col_d = st.columns(2)
    with col_c:
        semantic_top_k = st.number_input(
            "top_k (semantic)",
            min_value=1,
            max_value=200,
            value=int(active_search.get("semantic_top_k", 5)),
            step=1,
            key=f"{file_hash}_semantic_top_k",
        )
        semantic_threshold = st.slider(
            "threshold (semantic)",
            min_value=0.0,
            max_value=1.0,
            value=float(active_search.get("semantic_threshold", 0.5)),
            step=0.01,
            key=f"{file_hash}_semantic_threshold",
        )
        semantic_deduplicate = st.checkbox(
            "Удалять дубли (semantic)",
            value=bool(active_search.get("semantic_deduplicate", True)),
            key=f"{file_hash}_semantic_deduplicate",
        )
        enable_keyword = st.checkbox(
            "Включить точный поиск",
            value=bool(active_search.get("enable_keyword", True)),
            key=f"{file_hash}_enable_keyword",
        )
        keyword_deduplicate = st.checkbox(
            "Удалять дубли (keyword)",
            value=bool(active_search.get("keyword_deduplicate", True)),
            key=f"{file_hash}_keyword_deduplicate",
        )
    with col_d:
        title_options = ["phrase"] + display_cols
        current_title = active_ui.get("title_column", "phrase")
        if current_title not in title_options:
            current_title = title_options[0]
        title_column = st.selectbox(
            "Заголовок карточки",
            options=title_options,
            index=title_options.index(current_title),
            key=f"{file_hash}_title_column",
        )
        show_filters = st.checkbox(
            "Показывать фильтры",
            value=bool(active_ui.get("show_filters", True)),
            key=f"{file_hash}_show_filters",
        )
        show_comment = st.checkbox(
            "Показывать комментарий",
            value=bool(active_ui.get("show_comment", True)),
            key=f"{file_hash}_show_comment",
        )
        show_matched_phrase = st.checkbox(
            "Показывать match-фразу",
            value=bool(active_ui.get("show_matched_phrase", True)),
            key=f"{file_hash}_show_matched_phrase",
        )

    parse_profile = {
        "search": {
            "split_newline": search_split_newline,
            "split_pipe": search_split_pipe,
            "split_slash": search_split_slash,
        },
        "filter": {
            "split_newline": filter_split_newline,
            "split_pipe": filter_split_pipe,
        },
    }
    mapping = {
        "search_cols": search_cols,
        "filter_cols": filter_cols,
        "display_cols": display_cols,
        "comment_col": comment_col,
    }
    search_cfg = {
        "semantic_top_k": int(semantic_top_k),
        "semantic_threshold": float(semantic_threshold),
        "semantic_deduplicate": bool(semantic_deduplicate),
        "enable_keyword": bool(enable_keyword),
        "keyword_deduplicate": bool(keyword_deduplicate),
    }
    ui_cfg = {
        "title_column": title_column,
        "show_filters": bool(show_filters),
        "show_comment": bool(show_comment),
        "show_matched_phrase": bool(show_matched_phrase),
    }

    project_name_default = st.session_state.get("active_project_name", "Новый проект")
    project_name = st.text_input("Имя проекта", value=project_name_default, key=f"{file_hash}_project_name")

    config = _build_config(
        project_name=project_name,
        mapping=mapping,
        parse_profile=parse_profile,
        search_cfg=search_cfg,
        ui_cfg=ui_cfg,
    )

    col_e, col_f, col_g = st.columns(3)
    with col_e:
        if st.button("Собрать предпросмотр", type="primary", use_container_width=True):
            if not mapping["search_cols"]:
                st.error("Нужно выбрать хотя бы одну колонку поиска.")
            else:
                st.session_state["builder_config"] = config
                st.session_state["active_project_name"] = project_name
                st.success("Конфигурация применена. Откройте вкладку «Предпросмотр».")

    with col_f:
        if st.button("Сохранить проект", use_container_width=True):
            if not mapping["search_cols"]:
                st.error("Нельзя сохранить проект без search-колонок.")
            else:
                try:
                    _save_project(
                        project_name=project_name,
                        file_name=file_name,
                        file_bytes=file_bytes,
                        sheet_name=sheet_name,
                        config=config,
                    )
                except Exception as exc:
                    st.error(f"Не удалось сохранить проект: {exc}")
                else:
                    st.success("Проект сохранен.")

    with col_g:
        st.download_button(
            "Скачать config JSON",
            data=json.dumps(config, ensure_ascii=False, indent=2),
            file_name=f"{_slugify(project_name)}.json",
            mime="application/json",
            use_container_width=True,
        )


def _render_saved_projects_panel():
    with st.sidebar:
        st.markdown("### Проекты")
        manifests = _list_saved_projects()
        if not manifests:
            st.caption("Сохраненных проектов пока нет.")
            return

        options = {path.stem: path for path in manifests}
        selected = st.selectbox("Открыть сохраненный", options=list(options.keys()))
        if st.button("Загрузить проект", use_container_width=True):
            try:
                payload, file_bytes = _load_saved_project(options[selected])
            except Exception as exc:
                st.error(f"Ошибка загрузки проекта: {exc}")
                return

            source = payload.get("source", {})
            st.session_state["builder_file_name"] = source.get("file_name")
            st.session_state["builder_file_bytes"] = file_bytes
            st.session_state["builder_sheet_name"] = source.get("sheet_name")
            st.session_state["builder_config"] = payload.get("config", {})
            st.session_state["active_project_name"] = payload.get("project_name", selected)
            st.success(f"Проект «{selected}» загружен.")


st.set_page_config(page_title="Конструктор поиска по таблицам", layout="wide")
if not check_password():
    st.stop()

st.title("Конструктор поиска по таблицам")
st.caption(
    "MVP: загрузите таблицу, назначьте типы колонок, задайте параметры поиска и сразу протестируйте результат."
)

_render_saved_projects_panel()

tab_builder, tab_preview = st.tabs(["Конструктор", "Предпросмотр"])

with tab_builder:
    _render_builder()

with tab_preview:
    cfg = st.session_state.get("builder_config")
    file_name = st.session_state.get("builder_file_name")
    file_bytes = st.session_state.get("builder_file_bytes")
    sheet_name = st.session_state.get("builder_sheet_name")

    if not cfg or not file_name or not file_bytes:
        st.info("Сначала настройте проект на вкладке «Конструктор».")
    else:
        try:
            _render_preview(cfg, file_name, file_bytes, sheet_name)
        except Exception as exc:
            st.error(f"Ошибка предпросмотра: {exc}")
