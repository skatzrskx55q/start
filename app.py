import hashlib
import hmac
import html
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

# -----------------------------------------------------------------------------
#  Вспомогательные функции
#
#  Функция `highlight_terms` подсвечивает совпадения слов из запроса в строке,
#  оборачивая их тегом <mark>. Она используется при отрисовке карточек с
#  результатами, чтобы выделять найденные слова и повышать читаемость.

def highlight_terms(text: str, query: str) -> str:
    """Возвращает HTML‑строку с подсветкой слов из запроса.

    Каждое слово из query (разделитель — пробел) ищется в text и
    оборачивается тегом <mark>. Регистр игнорируется. При отсутствии
    запроса возвращается HTML‑экранированный исходный текст.
    """
    text = str(text)
    query = query.strip()
    if not query:
        return html.escape(text)
    # Разбиваем запрос на слова и экранируем для составления regex
    words = [re.escape(w) for w in query.split() if w]
    if not words:
        return html.escape(text)
    pattern = re.compile("|".join(words), re.IGNORECASE)
    parts = []
    last_end = 0
    for m in pattern.finditer(text):
        # Экранируем текст между совпадениями
        parts.append(html.escape(text[last_end:m.start()]))
        # Оборачиваем совпадение
        parts.append(f"<mark>{html.escape(m.group(0))}</mark>")
        last_end = m.end()
    parts.append(html.escape(text[last_end:]))
    return "".join(parts)

# Функция для разворачивания результатов поиска в таблицу
def flatten_results_to_df(results: list[dict], include_score: bool = False) -> pd.DataFrame:
    """Преобразует список результатов в DataFrame.

    Аргумент include_score указывает, добавлять ли колонку score. Словари
    `displays` и `filters` разворачиваются в отдельные колонки.
    """
    rows = []
    for item in results:
        row: dict[str, object] = {}
        if include_score and "score" in item:
            row["score"] = item.get("score", 0.0)
        row["phrase"] = item.get("phrase", "")
        # Разворачиваем отображаемые колонки
        for col, val in item.get("displays", {}).items():
            row[col] = val
        # Разворачиваем фильтровые колонки
        for col, val in item.get("filters", {}).items():
            row[col] = val
        row["comment"] = item.get("comment", "")
        rows.append(row)
    return pd.DataFrame(rows)

# -----------------------------------------------------------------------------
#  Пользовательские уведомления

def _build_alert_html(message: str, alert_type: str) -> str:
    """
    Возвращает HTML для уведомления с заданным типом.

    alert_type может быть 'info', 'warning', 'error' или 'success'.
    """
    # Определяем иконку для каждого типа уведомления
    icon_class = {
        "info": "fa-circle-info",
        "warning": "fa-triangle-exclamation",
        "error": "fa-circle-xmark",
        "success": "fa-circle-check",
    }.get(alert_type, "fa-circle-info")
    return (
        f"<div class='ux-alert {alert_type}'>"
        f"<i class='fa-solid {icon_class}'></i>"
        f"{html.escape(message)}"
        "</div>"
    )


def show_info(message: str) -> None:
    """Отображает информационное сообщение с современной иконкой."""
    st.markdown(_build_alert_html(message, "info"), unsafe_allow_html=True)


def show_warning(message: str) -> None:
    """Отображает предупреждающее сообщение с современной иконкой."""
    st.markdown(_build_alert_html(message, "warning"), unsafe_allow_html=True)


def show_error(message: str) -> None:
    """Отображает сообщение об ошибке с современной иконкой."""
    st.markdown(_build_alert_html(message, "error"), unsafe_allow_html=True)


def show_success(message: str) -> None:
    """Отображает успешное сообщение с современной иконкой."""
    st.markdown(_build_alert_html(message, "success"), unsafe_allow_html=True)

# -----------------------------------------------------------------------------
#  Компонент пошагового мастера

def _render_stepper(current_step: int, total_steps: int, titles: list[str], icons: list[str]) -> None:
    """Отрисовывает горизонтальный индикатор шагов (stepper).

    current_step — номер текущего шага (1-indexed), total_steps — общее количество шагов.
    titles и icons — списки названий и классов Font Awesome для каждого шага.
    """
    parts: list[str] = []
    for i in range(total_steps):
        step_index = i + 1
        active = "active" if step_index == current_step else ""
        icon = icons[i] if i < len(icons) else "fa-circle"
        title = titles[i] if i < len(titles) else f"Шаг {step_index}"
        parts.append(
            f"<div class='step {active}'>"
            f"<div class='circle'><i class='fa-solid {icon}'></i></div>"
            f"<div class='label'>{html.escape(title)}</div>"
            "</div>"
        )
    html_content = "<div class='stepper'>" + "".join(parts) + "</div>"
    st.markdown(html_content, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
#  Глобальные константы и настройки

PROJECTS_DIR = Path(".projects")
PROJECT_DATA_DIR = PROJECTS_DIR / "data"

# -----------------------------------------------------------------------------
#  Аутентификация

def check_password():
    expected = os.getenv("APP_PASSWORD")
    if not expected:
        # Показываем ошибку, если пароль не задан в окружении
        show_error("APP_PASSWORD не задан в окружении.")
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
    # Сообщение о возможной задержке
    show_info("После ввода верного пароля первый запуск может занять некоторое время. Пожалуйста, подождите.")
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        show_error("Неверный пароль")
    return False

# -----------------------------------------------------------------------------
#  Стили приложения

def _inject_custom_styles():
    """
    Вставляет кастомный CSS и подключает внешние шрифты и иконки.

    Этот метод загружает Google‑шрифт Inter и библиотеку Font Awesome для
    использования современных иконок. Также задаются переменные цветов и
    стили для карточек, чипов и сообщений.
    """
    st.markdown(
        """
        <!-- Подключаем Google Fonts и Font Awesome -->
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css" integrity="sha512-dBwexyBdSovczvKhrcGFWUybgoRfpcpiAKzCd15oBaeRVm2U9VMC4/q3FM9pu+UciOW25SDnZ8NS52kLZMOt9g==" crossorigin="anonymous" referrerpolicy="no-referrer" />

        <style>
        /* Глобальные переменные цвета и шрифта */
        :root {
            --ux-primary: #6366f1;
            --ux-secondary: #8b5cf6;
            --ux-bg: #f8fafc;
            --ux-surface: #ffffff;
            --ux-text: #1f2937;
            --ux-muted: #6b7280;
            --ux-border: #e5e7eb;
            --ux-chip-bg: #eef2ff;
            --ux-chip-text: #4f46e5;
            --ux-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        }

        html[data-theme="dark"] {
            --ux-primary: #8b5cf6;
            --ux-secondary: #6366f1;
            --ux-bg: #0f172a;
            --ux-surface: #1e293b;
            --ux-text: #f1f5f9;
            --ux-muted: #9ca3af;
            --ux-border: #374151;
            --ux-chip-bg: #312e81;
            --ux-chip-text: #a5b4fc;
            --ux-shadow: 0 2px 8px rgba(0, 0, 0, 0.5);
        }

        @media (prefers-color-scheme: dark) {
            :root {
                --ux-primary: #8b5cf6;
                --ux-secondary: #6366f1;
                --ux-bg: #0f172a;
                --ux-surface: #1e293b;
                --ux-text: #f1f5f9;
                --ux-muted: #9ca3af;
                --ux-border: #374151;
                --ux-chip-bg: #312e81;
                --ux-chip-text: #a5b4fc;
                --ux-shadow: 0 2px 8px rgba(0, 0, 0, 0.5);
            }
        }

        /* Основные настройки страницы */
        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(1300px 500px at 15% -10%, rgba(99, 102, 241, 0.12), transparent 55%),
                radial-gradient(1000px 500px at 95% 0%, rgba(139, 92, 246, 0.10), transparent 55%),
                var(--ux-bg);
        }

        [data-testid="stHeader"] { background: transparent; }
        .block-container { max-width: 1180px; padding-top: 1.15rem; }

        /* Стили для кнопок */
        .stButton > button, .stDownloadButton > button {
            border-radius: 12px;
            border: 1px solid var(--ux-border);
            font-weight: 600;
            padding: 0.5rem 1rem;
            transition: all 0.2s ease;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            border-color: var(--ux-primary);
            color: var(--ux-primary);
        }

        /* Стили для виджетов ввода */
        div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, textarea {
            border-radius: 12px !important;
        }

        /* Карточки результатов */
        .ux-card {
            background: var(--ux-surface);
            border: 1px solid var(--ux-border);
            border-radius: 16px;
            box-shadow: var(--ux-shadow);
            padding: 16px 18px;
            margin-bottom: 14px;
        }
        .ux-card-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            margin-bottom: 10px;
        }
        .ux-title-wrap {
            display: flex;
            align-items: center;
            gap: 10px;
            min-width: 0;
        }
        .ux-index {
            min-width: 28px;
            height: 28px;
            border-radius: 999px;
            border: 1px solid var(--ux-border);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            color: var(--ux-muted);
            font-size: 12px;
            font-weight: 700;
        }
        .ux-title {
            color: var(--ux-text);
            font-size: 18px;
            font-weight: 700;
            line-height: 1.3;
            overflow-wrap: anywhere;
        }
        .ux-score {
            color: var(--ux-muted);
            font-size: 12px;
            border: 1px solid var(--ux-border);
            border-radius: 999px;
            padding: 5px 10px;
            white-space: nowrap;
        }
        .ux-kv {
            margin-top: 8px;
            display: grid;
            grid-template-columns: 150px 1fr;
            gap: 8px 10px;
            align-items: start;
        }
        .ux-k {
            color: var(--ux-muted);
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 0.02em;
            text-transform: uppercase;
        }
        .ux-v {
            color: var(--ux-text);
            font-size: 14px;
            overflow-wrap: anywhere;
        }
        .ux-chip-row {
            margin-top: 10px;
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
        }
        .ux-chip {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            border: 1px solid var(--ux-border);
            background: var(--ux-chip-bg);
            color: var(--ux-chip-text);
            padding: 4px 10px;
            font-size: 12px;
            font-weight: 600;
            line-height: 1.2;
        }
        .ux-chip.phrase { background: transparent; color: var(--ux-text); }
        .ux-comment {
            margin-top: 10px;
            border: 1px dashed var(--ux-border);
            border-radius: 12px;
            padding: 10px 12px;
            color: var(--ux-text);
            font-size: 14px;
            line-height: 1.45;
            overflow-wrap: anywhere;
        }
        .ux-subsection { margin-top: 20px; }

        /* Стили для пользовательских уведомлений */
        .ux-alert {
            display: flex;
            align-items: center;
            border-radius: 12px;
            padding: 12px 16px;
            margin: 10px 0;
            font-size: 14px;
            font-weight: 500;
        }
        .ux-alert i {
            margin-right: 8px;
            font-size: 16px;
        }
        .ux-alert.info {
            background-color: rgba(99, 102, 241, 0.1);
            color: var(--ux-text);
        }
        .ux-alert.info i {
            color: var(--ux-primary);
        }
        .ux-alert.warning {
            background-color: rgba(251, 191, 36, 0.12);
            color: #854d0e;
        }
        .ux-alert.warning i {
            color: #eab308;
        }
        .ux-alert.error {
            background-color: rgba(252, 165, 165, 0.12);
            color: #b91c1c;
        }
        .ux-alert.error i {
            color: #ef4444;
        }
        .ux-alert.success {
            background-color: rgba(34, 197, 94, 0.12);
            color: #166534;
        }
        .ux-alert.success i {
            color: #22c55e;
        }
        /* Шаги мастера конфигурации */
        .stepper {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 1.5rem;
        }
        .step {
            position: relative;
            flex: 1;
            text-align: center;
        }
        .step .circle {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            border: 2px solid var(--ux-border);
            background: var(--ux-surface);
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--ux-muted);
            font-size: 18px;
        }
        .step.active .circle {
            background: var(--ux-primary);
            border-color: var(--ux-primary);
            color: #fff;
        }
        .step .label {
            margin-top: 0.5rem;
            font-size: 12px;
            color: var(--ux-muted);
            white-space: nowrap;
        }
        .step.active .label {
            color: var(--ux-primary);
            font-weight: 600;
        }
        .step:not(:last-child)::after {
            content: "";
            position: absolute;
            top: 20px;
            right: -50%;
            width: 100%;
            height: 2px;
            background: var(--ux-border);
        }
        .step.active + .step::after {
            background: var(--ux-primary);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
#  Вспомогательные утилиты для фильтров и мэппинга

def _slugify(value):
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")
    return slug or "project"

def _file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()

def _is_excel(file_name: str) -> bool:
    return Path(file_name).suffix.lower() in {".xlsx", ".xlsm", ".xls"}

@st.cache_data(show_spinner=False)
def _get_excel_sheets(file_bytes: bytes):
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    return xls.sheet_names

@st.cache_data(show_spinner=False)
def _read_table(file_bytes: bytes, file_name: str, sheet_name: str | None = None):
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

def _auto_prefixed_columns(columns: list[str], prefix: str) -> list[str]:
    pattern = re.compile(rf"^{re.escape(prefix.lower())}\d+$")
    matched = [c for c in columns if pattern.match(str(c).lower())]
    return sorted(matched, key=lambda name: int(re.findall(r"\d+$", name)[0]) if re.findall(r"\d+$", name) else 10**9)

def _auto_mapping(columns: list[str]) -> dict[str, object]:
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

def _merge_parse_profile(overrides: dict | None) -> dict:
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

def _split_filter_values(value: str, split_newline: bool = True, split_pipe: bool = True) -> list[str]:
    text = str(value).strip()
    if not text:
        return []
    chunks = text.split("\n") if split_newline else [text]
    result: list[str] = []
    for chunk in chunks:
        parts = chunk.split("|") if split_pipe else [chunk]
        for part in parts:
            part = part.strip()
            if part:
                result.append(part)
    return result

def _dedup_keep_order(items: list) -> list:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        key = str(item).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered

def _collect_filter_options(df_runtime: pd.DataFrame, col: str, filter_profile: dict) -> list[str]:
    options: set[str] = set()
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

def _has_filter_selection(selected_filters: dict[str, list[str]]) -> bool:
    return any(bool(values) for values in selected_filters.values())

def _row_matches_selected_filters(row: pd.Series, selected_filters: dict[str, list[str]], filter_profile: dict) -> bool:
    for col, selected_values in selected_filters.items():
        if not selected_values:
            continue
        raw_value = row.get(col, "")
        values = _split_filter_values(
            raw_value,
            split_newline=filter_profile.get("split_newline", True),
            split_pipe=filter_profile.get("split_pipe", True),
        )
        if not set(values) & set(selected_values):
            return False
    return True

def _result_matches_filters(result: dict, selected_filters: dict[str, list[str]], filter_profile: dict) -> bool:
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

def _safe_inline_text(value: object) -> str:
    return html.escape(str(value))

def _safe_multiline_text(value: object) -> str:
    return html.escape(str(value)).replace("\n", "<br>")

def _chips_html(items: list[str], class_name: str = "ux-chip") -> str:
    cleaned = _dedup_keep_order(items)
    if not cleaned:
        return ""
    return "".join(
        f"<span class='{class_name}'>{_safe_inline_text(item)}</span>"
        for item in cleaned
    )

def _filter_chips_from_dict(filters: dict[str, str], filter_profile: dict) -> list[str]:
    chips: list[str] = []
    for col, raw in filters.items():
        values = _split_filter_values(
            raw,
            split_newline=filter_profile.get("split_newline", True),
            split_pipe=filter_profile.get("split_pipe", True),
        )
        if not values and str(raw).strip():
            values = [str(raw).strip()]
        for value in values:
            chips.append(f"{col}: {value}")
    return _dedup_keep_order(chips)

def _build_config(project_name: str, mapping: dict, parse_profile: dict, search_cfg: dict, ui_cfg: dict) -> dict:
    return {
        "project_name": project_name,
        "mapping": mapping,
        "parse_profile": parse_profile,
        "search": search_cfg,
        "ui": ui_cfg,
    }

def _project_registry_path(project_slug: str) -> Path:
    return PROJECTS_DIR / f"{project_slug}.json"

def _save_project(project_name: str, file_name: str, file_bytes: bytes, sheet_name: str | None, config: dict) -> None:
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

def _list_saved_projects() -> list[Path]:
    if not PROJECTS_DIR.exists():
        return []
    return sorted(PROJECTS_DIR.glob("*.json"))

def _load_saved_project(project_manifest_path: Path) -> tuple[dict, bytes]:
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
def _build_runtime_df(file_hash: str, file_bytes: bytes, file_name: str, sheet_name: str | None, config_json: str) -> pd.DataFrame:
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

def _build_filter_groups(df_runtime: pd.DataFrame, mapping: dict, selected_filters: dict[str, list[str]], filter_profile: dict) -> list[dict]:
    original_map: dict[str, set] = df_runtime.attrs.get("original_examples_map", {})
    groups: dict[str, dict] = {}
    for _, row in df_runtime.iterrows():
        if not _row_matches_selected_filters(row, selected_filters, filter_profile):
            continue
        original_index = str(row.get("original_index", "")).strip()
        if not original_index or original_index in groups:
            continue
        displays = {col: str(row.get(col, "")).strip() for col in mapping["display_cols"]}
        filters = {col: str(row.get(col, "")).strip() for col in mapping["filter_cols"]}
        comment = str(row.get("comment1", "")).strip()
        all_phrases = sorted(original_map.get(original_index, set()))
        if not all_phrases:
            phrase = str(row.get("phrase", "")).strip()
            all_phrases = [phrase] if phrase else []
        groups[original_index] = {
            "displays": displays,
            "filters": filters,
            "comment": comment,
            "all_phrases": all_phrases,
            "phrase": all_phrases[0] if all_phrases else "",
        }
    return list(groups.values())

def _render_result_card(
    item: dict,
    idx: int,
    title_column: str,
    show_score: bool,
    show_comment: bool,
    show_phrase: bool,
    display_cols: list[str],
    filter_cols: list[str],
    filter_profile: dict,
    query: str = "",
):
    displays = item.get("displays", {})
    filters = item.get("filters", {})
    phrase = str(item.get("phrase", "")).strip()
    comment = str(item.get("comment", "")).strip()
    if title_column == "phrase":
        raw_title = phrase
    else:
        raw_title = str(displays.get(title_column, "")).strip() or phrase
    title_html = highlight_terms(raw_title, query)
    score_html = ""
    if show_score and "score" in item:
        score_html = f"<span class='ux-score'>Релевантность: {float(item['score']):.2f}</span>"
    kv_rows: list[str] = []
    if show_phrase and phrase:
        kv_rows.append(
            f"<div class='ux-k'>match</div><div class='ux-v'>{highlight_terms(phrase, query)}</div>"
        )
    for col in display_cols:
        value = str(displays.get(col, "")).strip()
        if not value or col == title_column:
            continue
        kv_rows.append(
            f"<div class='ux-k'>{_safe_inline_text(col)}</div><div class='ux-v'>{highlight_terms(value, query)}</div>"
        )
    kv_html = f"<div class='ux-kv'>{''.join(kv_rows)}</div>" if kv_rows else ""
    filter_chips = _filter_chips_from_dict(
        {col: filters.get(col, "") for col in filter_cols},
        filter_profile,
    )
    chips_html = _chips_html(filter_chips)
    chips_block = f"<div class='ux-chip-row'>{chips_html}</div>" if chips_html else ""
    comment_block = ""
    if show_comment and comment:
        comment_block = (
            "<div class='ux-comment'>"
            f"{_safe_multiline_text(comment)}"
            "</div>"
        )
    st.markdown(
        (
            "<div class='ux-card'>"
            "<div class='ux-card-top'>"
            "<div class='ux-title-wrap'>"
            f"<span class='ux-index'>{idx}</span>"
            f"<div class='ux-title'>{title_html}</div>"
            "</div>"
            f"{score_html}"
            "</div>"
            f"{kv_html}"
            f"{chips_block}"
            f"{comment_block}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

def _render_filter_group_card(
    group_item: dict,
    idx: int,
    title_column: str,
    show_comment: bool,
    display_cols: list[str],
    filter_cols: list[str],
    filter_profile: dict,
):
    displays = group_item.get("displays", {})
    filters = group_item.get("filters", {})
    comment = str(group_item.get("comment", "")).strip()
    phrases = [str(v).strip() for v in group_item.get("all_phrases", []) if str(v).strip()]
    if title_column == "phrase":
        title_text = phrases[0] if phrases else ""
    else:
        title_text = str(displays.get(title_column, "")).strip()
    if not title_text:
        title_text = phrases[0] if phrases else "Без названия"
    kv_rows: list[str] = []
    for col in display_cols:
        value = str(displays.get(col, "")).strip()
        if not value or col == title_column:
            continue
        kv_rows.append(
            f"<div class='ux-k'>{_safe_inline_text(col)}</div><div class='ux-v'>{_safe_multiline_text(value)}</div>"
        )
    kv_html = f"<div class='ux-kv'>{''.join(kv_rows)}</div>" if kv_rows else ""
    filter_chips = _filter_chips_from_dict(
        {col: filters.get(col, "") for col in filter_cols},
        filter_profile,
    )
    filter_chip_html = _chips_html(filter_chips)
    filter_block = f"<div class='ux-chip-row'>{filter_chip_html}</div>" if filter_chip_html else ""
    phrase_chip_html = _chips_html(phrases, class_name="ux-chip phrase")
    phrase_block = f"<div class='ux-chip-row'>{phrase_chip_html}</div>" if phrase_chip_html else ""
    comment_block = ""
    if show_comment and comment:
        comment_block = (
            "<div class='ux-comment'>"
            f"{_safe_multiline_text(comment)}"
            "</div>"
        )
    st.markdown(
        (
            "<div class='ux-card'>"
            "<div class='ux-card-top'>"
            "<div class='ux-title-wrap'>"
            f"<span class='ux-index'>{idx}</span>"
            f"<div class='ux-title'>{_safe_multiline_text(title_text)}</div>"
            "</div>"
            "</div>"
            f"{kv_html}"
            f"{filter_block}"
            f"{phrase_block}"
            f"{comment_block}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

def _render_preview(config: dict, file_name: str, file_bytes: bytes, sheet_name: str | None):
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
    mapping = config.get("mapping", {})
    parse_profile = _merge_parse_profile(config.get("parse_profile"))
    search_cfg = config.get("search", {})
    ui_cfg = config.get("ui", {})
    mapping.setdefault("search_cols", [])
    mapping.setdefault("filter_cols", [])
    mapping.setdefault("display_cols", [])
    st.caption(
        f"Строк после развертки search: {len(df_runtime)} | "
        f"search: {len(mapping['search_cols'])}, "
        f"filter: {len(mapping['filter_cols'])}, "
        f"display: {len(mapping['display_cols'])}"
    )
    # ----------- Блок фильтров -----------
    # Собираем выбранные значения фильтров в словарь. Если фильтры включены в настройках
    # и есть хотя бы одна фильтровая колонка, отображаем панель фильтров в виде экспандера.
    selected_filters: dict[str, list[str]] = {}
    if ui_cfg.get("show_filters", True) and mapping["filter_cols"]:
        with st.expander("Фильтр по тематикам", expanded=False):
            for col in mapping["filter_cols"]:
                options = _collect_filter_options(df_runtime, col, parse_profile["filter"])
                key = f"{file_hash}_filter_{col}"
                selected_filters[col] = st.multiselect(col, options=options, key=key)

    # Если выбраны хотя бы какие‑то фильтры, отображаем результаты фильтрации
    if mapping["filter_cols"] and _has_filter_selection(selected_filters):
        st.markdown("<div class='ux-subsection'></div>", unsafe_allow_html=True)
        st.markdown("### Результаты фильтра")
        groups = _build_filter_groups(
            df_runtime,
            mapping,
            selected_filters,
            parse_profile["filter"],
        )
        if groups:
            for idx, group in enumerate(groups, start=1):
                _render_filter_group_card(
                    group_item=group,
                    idx=idx,
                    title_column=ui_cfg.get("title_column", "phrase"),
                    show_comment=bool(ui_cfg.get("show_comment", True)),
                    display_cols=mapping["display_cols"],
                    filter_cols=mapping["filter_cols"],
                    filter_profile=parse_profile["filter"],
                )
        else:
            show_warning("По выбранным тематикам ничего не найдено.")
    # ----------- Блок поиска -----------
    st.markdown("<div class='ux-subsection'></div>", unsafe_allow_html=True)
    st.markdown("### Поиск")
    allow_keyword = bool(search_cfg.get("enable_keyword", True))
    mode_options = ["Умный"]
    if allow_keyword:
        mode_options.extend(["Точный", "Оба"])
    default_mode = search_cfg.get("default_mode", "Умный")
    if default_mode not in mode_options:
        default_mode = mode_options[0]
    search_mode = st.radio(
        "Режим поиска",
        options=mode_options,
        index=mode_options.index(default_mode),
        horizontal=True,
        key=f"{file_hash}_search_mode",
    )
    apply_filters_to_search = False
    if mapping["filter_cols"] and _has_filter_selection(selected_filters):
        apply_filters_to_search = st.checkbox(
            "Применять выбранные фильтры к поиску",
            value=bool(search_cfg.get("apply_filters_to_search_default", False)),
            key=f"{file_hash}_apply_filters_to_search",
        )
    query_key = f"{file_hash}_query_input"
    last_query_key = f"{file_hash}_last_query"
    with st.form(key=f"{file_hash}_search_form"):
        # Кастомный поисковый инпут с иконкой поиска
        search_icon_col, search_input_col = st.columns([1, 9])
        with search_icon_col:
            st.markdown(
                "<i class='fa-solid fa-magnifying-glass' style='font-size:20px; color: var(--ux-muted); padding-top: 0.5rem;'></i>",
                unsafe_allow_html=True,
            )
        with search_input_col:
            query_input = st.text_input("", placeholder="Введите запрос", key=query_key)
        submitted = st.form_submit_button("Найти", use_container_width=True)
    if submitted:
        st.session_state[last_query_key] = query_input.strip()
    query = st.session_state.get(last_query_key, "").strip()
    if not query:
        show_info("Введите запрос и нажмите «Найти».")
        return
    # Выводим активные фильтры, если они выбраны
    if _has_filter_selection(selected_filters):
        active_filter_chips: list[str] = []
        for col, values in selected_filters.items():
            for val in values:
                active_filter_chips.append(f"{col}: {val}")
        chips_html = _chips_html(active_filter_chips)
        st.markdown("<div class='ux-subsection'></div>", unsafe_allow_html=True)
        st.markdown("### Активные фильтры")
        st.markdown(f"<div class='ux-chip-row'>{chips_html}</div>", unsafe_allow_html=True)
    should_apply_filters_to_search = apply_filters_to_search and _has_filter_selection(selected_filters)
    # ------------- Умный поиск -------------
    if search_mode in ("Умный", "Оба"):
        semantic_results = semantic_search_rows(
            query,
            df_runtime,
            threshold=float(search_cfg.get("semantic_threshold", 0.5)),
            top_k=int(search_cfg.get("semantic_top_k", 5)),
            filter_cols=mapping["filter_cols"],
            display_cols=mapping["display_cols"],
            comment_col="comment1",
            deduplicate=bool(search_cfg.get("semantic_deduplicate", True)),
        )
        if should_apply_filters_to_search:
            semantic_results = [
                item
                for item in semantic_results
                if _result_matches_filters(item, selected_filters, parse_profile["filter"])
            ]
        st.markdown("### Умный поиск")
        if semantic_results:
            for idx, item in enumerate(semantic_results, start=1):
                _render_result_card(
                    item=item,
                    idx=idx,
                    title_column=ui_cfg.get("title_column", "phrase"),
                    show_score=True,
                    show_comment=bool(ui_cfg.get("show_comment", True)),
                    show_phrase=bool(ui_cfg.get("show_matched_phrase", True)),
                    display_cols=mapping["display_cols"],
                    filter_cols=mapping["filter_cols"],
                    filter_profile=parse_profile["filter"],
                    query=query,
                )
            # Кнопка скачивания для умного поиска
            df_sem = flatten_results_to_df(semantic_results, include_score=True)
            if not df_sem.empty:
                csv_sem = df_sem.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Скачать результаты умного поиска",
                    data=csv_sem,
                    file_name="semantic_results.csv",
                    mime="text/csv",
                )
        else:
            show_warning("Совпадений не найдено.")
    # ------------- Точный поиск -------------
    if allow_keyword and search_mode in ("Точный", "Оба"):
        keyword_results = keyword_search_rows(
            query,
            df_runtime,
            filter_cols=mapping["filter_cols"],
            display_cols=mapping["display_cols"],
            comment_col="comment1",
            deduplicate=bool(search_cfg.get("keyword_deduplicate", True)),
        )
        if should_apply_filters_to_search:
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
                    title_column=ui_cfg.get("title_column", "phrase"),
                    show_score=False,
                    show_comment=bool(ui_cfg.get("show_comment", True)),
                    show_phrase=bool(ui_cfg.get("show_matched_phrase", True)),
                    display_cols=mapping["display_cols"],
                    filter_cols=mapping["filter_cols"],
                    filter_profile=parse_profile["filter"],
                    query=query,
                )
            # Кнопка скачивания для точного поиска
            df_kw = flatten_results_to_df(keyword_results, include_score=False)
            if not df_kw.empty:
                csv_kw = df_kw.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Скачать результаты точного поиска",
                    data=csv_kw,
                    file_name="keyword_results.csv",
                    mime="text/csv",
                )
        else:
            show_info("Совпадений в точном поиске нет.")

def _render_builder_wizard() -> None:
    """Отображает пошаговый конструктор проекта.

    Мастер разделён на четыре шага: загрузка, назначение типов колонок, настройки,
    предпросмотр. Состояние текущего шага хранится в st.session_state['builder_step'].
    При переходе между шагами сохраняем промежуточные данные (файл, мэппинг,
    параметры).
    """
    # Определяем названия и иконки шагов для отображения в индикаторе
    step_titles = ["Загрузка", "Назначение", "Настройки", "Предпросмотр"]
    step_icons = ["fa-upload", "fa-table-cells", "fa-sliders-h", "fa-search"]
    total_steps = len(step_titles)
    # Получаем текущий шаг (1 по умолчанию)
    current_step = int(st.session_state.get("builder_step", 1))
    # Ограничиваем диапазон шагов
    if current_step < 1:
        current_step = 1
    if current_step > total_steps:
        current_step = total_steps
    st.session_state["builder_step"] = current_step
    # Отрисовываем индикатор прогресса
    _render_stepper(current_step, total_steps, step_titles, step_icons)
    # Читаем из session_state ранее загруженные данные
    file_name: str | None = st.session_state.get("builder_file_name")
    file_bytes: bytes | None = st.session_state.get("builder_file_bytes")
    sheet_name: str | None = st.session_state.get("builder_sheet_name")
    builder_mapping: dict | None = st.session_state.get("builder_mapping")
    builder_settings: dict | None = st.session_state.get("builder_settings")
    # Шаг 1: Загрузка файла
    if current_step == 1:
        st.header("Шаг 1: Загрузка таблицы")
        uploaded_file = st.file_uploader(
            "Выберите файл (Excel, CSV, TXT)",
            type=["xlsx", "xls", "xlsm", "csv", "txt"],
            key="_builder_file_uploader",
        )
        if uploaded_file is not None:
            file_name = uploaded_file.name
            file_bytes = uploaded_file.getvalue()
            # Сохраняем файл в session_state
            st.session_state["builder_file_name"] = file_name
            st.session_state["builder_file_bytes"] = file_bytes
            # Сбросить sheet_name при загрузке нового файла
            if not _is_excel(file_name):
                st.session_state["builder_sheet_name"] = None
                sheet_name = None
        # Если файл загружен, отображаем таблицу и выбираем лист Excel при необходимости
        if file_name and file_bytes:
            # Для Excel запрашиваем имя листа
            if _is_excel(file_name):
                try:
                    sheets = _get_excel_sheets(file_bytes)
                except Exception as exc:
                    show_error(f"Ошибка чтения Excel: {exc}")
                    sheets = []
                if sheets:
                    default_sheet = sheet_name if sheet_name in sheets else sheets[0]
                    sheet_name = st.selectbox(
                        "Лист Excel",
                        options=sheets,
                        index=sheets.index(default_sheet) if default_sheet in sheets else 0,
                        key="_builder_sheet_select",
                    )
                    st.session_state["builder_sheet_name"] = sheet_name
            # Читаем исходную таблицу для предпросмотра
            try:
                df_source = _read_table(file_bytes, file_name, sheet_name=sheet_name)
                st.caption(
                    f"Источник: `{file_name}` | Строк: {len(df_source)} | Колонок: {len(df_source.columns)}"
                )
                st.dataframe(df_source.head(20), use_container_width=True)
            except Exception as exc:
                show_error(f"Ошибка чтения таблицы: {exc}")
                df_source = None
            # Кнопка "Далее" активна только если данные успешно прочитаны
            col_prev, col_next = st.columns([1, 1])
            with col_prev:
                pass  # На первом шаге кнопка "Назад" не нужна
            with col_next:
                if st.button("Далее", type="primary", use_container_width=True):
                    if df_source is None:
                        show_error("Сначала исправьте ошибки чтения файла.")
                    else:
                        st.session_state["builder_step"] = 2
                        st.experimental_rerun()
        else:
            show_info("Загрузите таблицу, чтобы продолжить.")
        return
    # Шаг 2: Назначение типов колонок
    if current_step == 2:
        st.header("Шаг 2: Назначение типов колонок")
        # Если файл не загружен, возвращаемся к шагу 1
        if not file_name or not file_bytes:
            show_info("Сначала загрузите файл на предыдущем шаге.")
            if st.button("Назад", use_container_width=True):
                st.session_state["builder_step"] = 1
                st.experimental_rerun()
            return
        # Загружаем исходную таблицу (повторяем чтение при необходимости)
        try:
            df_source = _read_table(file_bytes, file_name, sheet_name=sheet_name)
        except Exception as exc:
            show_error(f"Ошибка чтения таблицы: {exc}")
            if st.button("Назад", use_container_width=True):
                st.session_state["builder_step"] = 1
                st.experimental_rerun()
            return
        columns = [str(col) for col in df_source.columns]
        auto = _auto_mapping(columns)
        # Получаем ранее выбранные мэппинги, если есть
        active_mapping = builder_mapping if builder_mapping else {}
        search_default = [c for c in active_mapping.get("search_cols", auto.get("search_cols", [])) if c in columns]
        filter_default = [c for c in active_mapping.get("filter_cols", auto.get("filter_cols", [])) if c in columns]
        display_default = [c for c in active_mapping.get("display_cols", auto.get("display_cols", [])) if c in columns]
        comment_default = active_mapping.get("comment_col", auto.get("comment_col"))
        # Выбор колонок
        search_cols = st.multiselect(
            "Колонки поиска (обязательно)",
            options=columns,
            default=search_default,
            key="_builder_search_cols",
        )
        filter_cols = st.multiselect(
            "Колонки фильтров (опционально)",
            options=columns,
            default=filter_default,
            key="_builder_filter_cols",
        )
        display_cols = st.multiselect(
            "Колонки вывода (опционально)",
            options=columns,
            default=display_default,
            key="_builder_display_cols",
        )
        comment_options = ["<нет>"] + columns
        if comment_default not in comment_options:
            comment_default = "<нет>"
        comment_col = st.selectbox(
            "Колонка комментария",
            options=comment_options,
            index=comment_options.index(comment_default),
            key="_builder_comment_col",
        )
        comment_col = None if comment_col == "<нет>" else comment_col
        # Кнопки управления
        col_prev, col_next = st.columns(2)
        with col_prev:
            if st.button("Назад", use_container_width=True):
                st.session_state["builder_step"] = 1
                st.experimental_rerun()
        with col_next:
            if st.button("Далее", type="primary", use_container_width=True):
                if not search_cols:
                    show_error("Выберите хотя бы одну колонку поиска.")
                else:
                    st.session_state["builder_mapping"] = {
                        "search_cols": list(search_cols),
                        "filter_cols": list(filter_cols),
                        "display_cols": list(display_cols),
                        "comment_col": comment_col,
                    }
                    st.session_state["builder_step"] = 3
                    st.experimental_rerun()
        return
    # Шаг 3: Настройки разбиения и поиска
    if current_step == 3:
        st.header("Шаг 3: Настройки разбиения и поиска")
        # Проверяем наличие мэппинга
        if not builder_mapping or not builder_mapping.get("search_cols"):
            show_error("Назначьте типы колонок на предыдущем шаге.")
            if st.button("Назад", use_container_width=True):
                st.session_state["builder_step"] = 2
                st.experimental_rerun()
            return
        # Загружаем активные настройки, если они были сохранены
        active_parse = {}
        active_search = {}
        active_ui = {}
        project_name_default = st.session_state.get("active_project_name", "Новый проект")
        if builder_settings:
            active_parse = builder_settings.get("parse_profile", {})
            active_search = builder_settings.get("search", {})
            active_ui = builder_settings.get("ui", {})
            project_name_default = builder_settings.get("project_name", project_name_default)
        # Раздел "Параметры разбиения"
        st.subheader("3.1 Разбиение поисковых и фильтровых колонок")
        merged_parse_profile = _merge_parse_profile(active_parse)
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Поисковые колонки**")
            search_split_newline = st.checkbox(
                "split `\\n`",
                value=merged_parse_profile["search"].get("split_newline", True),
                key="_builder_search_split_newline",
            )
            search_split_pipe = st.checkbox(
                "split `|`",
                value=merged_parse_profile["search"].get("split_pipe", True),
                key="_builder_search_split_pipe",
            )
            search_split_slash = st.checkbox(
                "split `/`",
                value=merged_parse_profile["search"].get("split_slash", True),
                key="_builder_search_split_slash",
            )
        with col_b:
            st.markdown("**Фильтровые колонки**")
            filter_split_newline = st.checkbox(
                "split `\\n` (filter)",
                value=merged_parse_profile["filter"].get("split_newline", True),
                key="_builder_filter_split_newline",
            )
            filter_split_pipe = st.checkbox(
                "split `|` (filter)",
                value=merged_parse_profile["filter"].get("split_pipe", True),
                key="_builder_filter_split_pipe",
            )
        # Раздел "Настройки поиска и UI"
        st.subheader("3.2 Настройки поиска и интерфейса")
        col_c, col_d = st.columns(2)
        with col_c:
            semantic_top_k = st.number_input(
                "top_k (semantic)",
                min_value=1,
                max_value=200,
                value=int(active_search.get("semantic_top_k", 5)),
                step=1,
                key="_builder_semantic_top_k",
            )
            semantic_threshold = st.slider(
                "threshold (semantic)",
                min_value=0.0,
                max_value=1.0,
                value=float(active_search.get("semantic_threshold", 0.5)),
                step=0.01,
                key="_builder_semantic_threshold",
            )
            semantic_deduplicate = st.checkbox(
                "Удалять дубли (semantic)",
                value=bool(active_search.get("semantic_deduplicate", True)),
                key="_builder_semantic_deduplicate",
            )
            enable_keyword = st.checkbox(
                "Разрешить точный поиск",
                value=bool(active_search.get("enable_keyword", True)),
                key="_builder_enable_keyword",
            )
            keyword_deduplicate = st.checkbox(
                "Удалять дубли (keyword)",
                value=bool(active_search.get("keyword_deduplicate", True)),
                key="_builder_keyword_deduplicate",
                disabled=not enable_keyword,
            )
            mode_options = ["Умный"] if not enable_keyword else ["Умный", "Точный", "Оба"]
            default_mode_current = active_search.get("default_mode", "Умный")
            if default_mode_current not in mode_options:
                default_mode_current = mode_options[0]
            default_mode = st.selectbox(
                "Режим поиска по умолчанию",
                options=mode_options,
                index=mode_options.index(default_mode_current),
                key="_builder_default_mode",
            )
            apply_filters_to_search_default = st.checkbox(
                "По умолчанию применять фильтры к поиску",
                value=bool(active_search.get("apply_filters_to_search_default", False)),
                key="_builder_apply_filters_to_search_default",
            )
        with col_d:
            # Определяем доступные варианты заголовка карточки
            display_cols = builder_mapping.get("display_cols", [])
            title_options = ["phrase"] + display_cols
            current_title = active_ui.get("title_column", "phrase")
            if current_title not in title_options:
                current_title = title_options[0]
            title_column = st.selectbox(
                "Заголовок карточки",
                options=title_options,
                index=title_options.index(current_title),
                key="_builder_title_column",
            )
            show_filters = st.checkbox(
                "Показывать фильтры",
                value=bool(active_ui.get("show_filters", True)),
                key="_builder_show_filters",
            )
            show_comment = st.checkbox(
                "Показывать комментарий",
                value=bool(active_ui.get("show_comment", True)),
                key="_builder_show_comment",
            )
            show_matched_phrase = st.checkbox(
                "Показывать match-фразу",
                value=bool(active_ui.get("show_matched_phrase", True)),
                key="_builder_show_matched_phrase",
            )
        project_name = st.text_input(
            "Имя проекта",
            value=project_name_default,
            key="_builder_project_name",
        )
        # Кнопки управления
        col_prev, col_next = st.columns(2)
        with col_prev:
            if st.button("Назад", use_container_width=True):
                st.session_state["builder_step"] = 2
                st.experimental_rerun()
        with col_next:
            if st.button("Далее", type="primary", use_container_width=True):
                # Сохраняем настройки в session_state
                st.session_state["builder_settings"] = {
                    "project_name": project_name,
                    "parse_profile": {
                        "search": {
                            "split_newline": bool(search_split_newline),
                            "split_pipe": bool(search_split_pipe),
                            "split_slash": bool(search_split_slash),
                        },
                        "filter": {
                            "split_newline": bool(filter_split_newline),
                            "split_pipe": bool(filter_split_pipe),
                        },
                    },
                    "search": {
                        "semantic_top_k": int(semantic_top_k),
                        "semantic_threshold": float(semantic_threshold),
                        "semantic_deduplicate": bool(semantic_deduplicate),
                        "enable_keyword": bool(enable_keyword),
                        "keyword_deduplicate": bool(keyword_deduplicate),
                        "default_mode": default_mode,
                        "apply_filters_to_search_default": bool(apply_filters_to_search_default),
                    },
                    "ui": {
                        "title_column": title_column,
                        "show_filters": bool(show_filters),
                        "show_comment": bool(show_comment),
                        "show_matched_phrase": bool(show_matched_phrase),
                    },
                }
                st.session_state["active_project_name"] = project_name
                st.session_state["builder_step"] = 4
                st.experimental_rerun()
        return
    # Шаг 4: Предпросмотр и поиск
    if current_step == 4:
        st.header("Шаг 4: Предпросмотр и поиск")
        # Проверяем, что файл и настройки есть
        if not file_name or not file_bytes or not builder_mapping or not builder_settings:
            show_error("Для предпросмотра необходимо выполнить предыдущие шаги.")
            if st.button("Назад", use_container_width=True):
                st.session_state["builder_step"] = 3
                st.experimental_rerun()
            return
        # Собираем конфигурацию проекта
        project_name = builder_settings.get("project_name", "Проект")
        parse_profile = builder_settings.get("parse_profile", {})
        search_cfg = builder_settings.get("search", {})
        ui_cfg = builder_settings.get("ui", {})
        config = _build_config(
            project_name=project_name,
            mapping=builder_mapping,
            parse_profile=parse_profile,
            search_cfg=search_cfg,
            ui_cfg=ui_cfg,
        )
        # Сохраняем активную конфигурацию в session_state для быстрого доступа
        st.session_state["builder_config"] = config
        # Кнопки сохранения и загрузки
        col_prev, col_save, col_download = st.columns([1, 1, 1])
        with col_prev:
            if st.button("Назад", use_container_width=True):
                st.session_state["builder_step"] = 3
                st.experimental_rerun()
        with col_save:
            if st.button("Сохранить проект", use_container_width=True):
                try:
                    _save_project(
                        project_name=project_name,
                        file_name=file_name,
                        file_bytes=file_bytes,
                        sheet_name=sheet_name,
                        config=config,
                    )
                except Exception as exc:
                    show_error(f"Не удалось сохранить проект: {exc}")
                else:
                    show_success("Проект сохранен.")
        with col_download:
            st.download_button(
                "Скачать config JSON",
                data=json.dumps(config, ensure_ascii=False, indent=2),
                file_name=f"{_slugify(project_name)}.json",
                mime="application/json",
                use_container_width=True,
            )
        # Выводим предпросмотр
        try:
            _render_preview(config, file_name, file_bytes, sheet_name)
        except Exception as exc:
            show_error(f"Ошибка предпросмотра: {exc}")
        return

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
                show_error(f"Ошибка загрузки проекта: {exc}")
                return
            source = payload.get("source", {})
            st.session_state["builder_file_name"] = source.get("file_name")
            st.session_state["builder_file_bytes"] = file_bytes
            st.session_state["builder_sheet_name"] = source.get("sheet_name")
            st.session_state["builder_config"] = payload.get("config", {})
            st.session_state["active_project_name"] = payload.get("project_name", selected)
            show_success(f"Проект «{selected}» загружен.")

st.set_page_config(page_title="Конструктор поиска по таблицам", layout="wide")
if not check_password():
    st.stop()
_inject_custom_styles()
st.title("Конструктор поиска по таблицам")
st.caption(
    "Этот мастер поможет загрузить таблицу, назначить типы колонок, настроить параметры и получить предпросмотр поиска."  # fmt: off
)
_render_saved_projects_panel()

# Отображаем пошаговый мастер
_render_builder_wizard()
