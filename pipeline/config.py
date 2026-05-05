"""
config.py — embedding model, relevance queries, and chunk size constants.

All chunk/token sizing is derived from MAX_CHUNK_TOKENS so chunker and
embedder always stay in sync. Change that one constant to resize everything.
"""

# ── Embedding model ───────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen3-Embedding-8B"

# Prepended to queries at inference time (not to documents).
QUERY_INSTRUCTION = "Instruct: Retrieve relevant university information for prospective students\nQuery: "

# ── Chunk / token sizing ──────────────────────────────────────────────────────
#
# MAX_CHUNK_TOKENS is the master constant.
# Chunker targets this size; embedder uses it as max_length.
#
# Memory per batch = batch_size × MAX_CHUNK_TOKENS × 2 bytes (fp16) × hidden_dim
# Qwen3-8B int4, MAX_CHUNK_TOKENS=1024, batch_size=8 ≈ ~2GB activation memory
#
# Word ↔ token ratio for Slavic text: ~1.35 tokens/word
#   600 words ≈ 810 tokens  → MAX_CHUNK_TOKENS=1024 covers all chunks with headroom
#   400 words ≈ 540 tokens  → MAX_CHUNK_TOKENS=640 is sufficient if you use smaller chunks

MAX_CHUNK_TOKENS = 640  # embedder max_length — set equal to or above chunk target

# Chunker targets this many tokens per chunk (≈ MAX_CHUNK_TOKENS * 0.75 to leave overlap room)
CHUNK_TARGET_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 64

# Word bounds for normalisation pass
CHUNK_MIN_WORDS = 40  # drop stubs
CHUNK_MAX_WORDS = int(CHUNK_TARGET_TOKENS / 1.35)

# ── Relevance queries ─────────────────────────────────────────────────────────

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
