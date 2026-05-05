"""
config.py — embedding model, relevance queries, and chunk size constants.

All chunk/token sizing is derived from MAX_CHUNK_TOKENS so chunker and
embedder always stay in sync. Change that one constant to resize everything.
"""

MODEL_ID = "Qwen/Qwen3-Embedding-8B"

QUERY_INSTRUCTION = "Instruct: Retrieve relevant university information for prospective students\nQuery: "
MAX_CHUNK_TOKENS = 640
CHUNK_TARGET_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 64
CHUNK_MIN_WORDS = 40
CHUNK_MAX_WORDS = int(CHUNK_TARGET_TOKENS / 1.35)

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
