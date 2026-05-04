"""
config.py — embedding model, relevance queries, and chunk size constants.
Edit this file to change what "relevant for applicants" means.
"""

# ── Embedding model ───────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen3-Embedding-8B"

# Prepended to queries at inference time (not to documents).
# Qwen3-Embedding is instruction-tuned — this prefix improves retrieval quality.
QUERY_INSTRUCTION = "Instruct: Retrieve relevant university information for prospective students\nQuery: "

# ── Relevance queries ─────────────────────────────────────────────────────────
# Each chunk is scored against ALL queries; max score is kept.
# Add more queries to broaden coverage, remove to narrow focus.

RELEVANCE_QUERIES = [
    "поступление в университет требования для абитуриентов",
    "факультеты специальности направления обучения бакалавриат магистратура",
    "условия проживания общежитие студенты",
    "стоимость обучения платное бесплатное стипендия финансовая помощь",
    "студенческая жизнь мероприятия кружки секции",
    "расписание занятий учебный план семестр экзамены",
    "кафедра научная деятельность преподаватели профессора",
    "международное сотрудничество обмен студентами программы",
    "admission university requirements applicants enrollment",
    "dormitory accommodation student housing",
]

# ── Chunk size bounds ─────────────────────────────────────────────────────────

CHUNK_MIN_WORDS = 40  # drop stubs: lone headings, single sentences
CHUNK_MAX_WORDS = 600  # re-split oversized chunks for uniform retrieval
