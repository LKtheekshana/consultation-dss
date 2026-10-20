"""Project settings shared by all stages.

Change values here instead of editing the scripts.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# Input file
# ---------------------------------------------------------------------------
# The real questionnaire export. In Google Forms: Responses > ⋮ > Download
# responses (.csv); unzip it and save the CSV in data/raw/ under this name.
# (The synthetic test file used while building the pipeline was deleted
# on 9 October 2026.)
INPUT_FILE = RAW_DIR / "form_responses.csv"

# ---------------------------------------------------------------------------
# Mapping from Google Form question titles to short column names.
# The left side must match the question titles in the form export exactly.
# ---------------------------------------------------------------------------
FORM_COLUMN_MAP = {
    "Timestamp": "timestamp",
    "Do you consent to participate in this study on the terms described above?": "consent",
    "Age range": "age_range",
    "Gender": "gender",
    "Province of residence": "province",
    "Have you previously contacted or used a government information center or "
    "consultation service in Sri Lanka (e.g. to report an issue, seek information, "
    "or give feedback)?": "prior_use",
    "If yes, how satisfied were you with that experience?": "satisfaction",
    "Which area does your issue relate to?": "issue_area",
    "Please name the area your issue relates to": "other_area",
    "Describe the problem or issue you have experienced, in as much detail as possible.": "problem_description",
    "What improvements or solutions would you suggest to the government regarding this issue?": "suggested_improvement",
    "In 2–3 words, what keywords would you use to summarize your issue above?": "keywords",
}

# Every domain page in the form asks this same checklist question, so the
# export contains one column per domain with this title. They are merged
# into a single "issues_selected" column.
ISSUES_QUESTION = "Which of these issues have you experienced? (select all that apply)"

# Answer that means the person did not consent (their row is removed)
CONSENT_NO = "No, I do not wish to participate"

# ---------------------------------------------------------------------------
# Cleaning rules
# ---------------------------------------------------------------------------
# Responses whose combined text has fewer words than this are removed as
# too short to be useful. Check the preprocessing report before changing it.
MIN_WORDS = 8

# T5 reads at most 512 tokens. Roughly 1 token is about 0.75 English words,
# so responses longer than this are flagged in the report.
T5_TOKEN_LIMIT = 512

# Responses removed by the researcher after reading them, with the reason.
# The reasons are logged in removed_responses.csv and the report. Keyed by
# the form's Timestamp, because response IDs (R001...) depend on row order.
MANUAL_EXCLUSIONS = {
    "2026/10/09 12:53:03 pm GMT+5:30": "off-topic (not a consultation submission)",
    "2026/10/09 2:08:34 pm GMT+5:30": "near-duplicate (reworded copy of another response)",
}

# Responses in Sinhala or Tamil written in English letters ("Singlish"),
# which the script-based language check sees as English.
FORCE_LANGUAGE = {
    "2026/10/09 12:39:27 pm GMT+5:30": "sinhala",
}

# English translations of Sinhala/Tamil responses (see README). A translation
# is used only after a person has checked it against the original and filled
# in "checked_by". Set to False only for trial runs, never for reported results.
TRANSLATIONS_FILE = RAW_DIR / "translations.csv"
TRANSLATIONS_REQUIRE_CHECK = True

# ---------------------------------------------------------------------------
# Stage 2: embeddings
# ---------------------------------------------------------------------------
# Sentence transformer used for all embeddings (responses and queries).
# It is downloaded automatically (~420 MB) the first time Stage 2 runs.
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# How many sentences are encoded at once. Lower it if the laptop runs out
# of memory.
EMBED_BATCH_SIZE = 32

EMBEDDINGS_FILE = PROCESSED_DIR / "embeddings.pkl"

# ---------------------------------------------------------------------------
# Stage 3: retrieval and thresholds
# ---------------------------------------------------------------------------
# Static method: a response is retrieved if its cosine similarity to the
# query is at least this value. 0.5 is a common starting point for
# all-mpnet-base-v2; the grid search shows whether a different value is better.
STATIC_THRESHOLD = 0.50

# Dynamic method: T = mean + k * std of the similarity scores for each query.
# k = 1.0 keeps responses that are clearly above average for that query.
# A negative k gives the "mean - k*std" variant.
DYNAMIC_K = 1.0

# Top-k method: always return the k most similar responses.
TOP_K = 5

# Web app: warn when even the best match is weak. The dynamic threshold always
# returns the responses that are most similar *relative to the others*, so vague
# keywords still get results. 0.37 is the threshold at which F1 peaked when
# "relevant" meant sharing a specific issue (Stage 3, Figure 5.2b): below it a
# response rarely concerns the same issue. Only a message; retrieval is unchanged.
WEAK_MATCH_SIMILARITY = 0.37

# Grid search: thresholds from 0 to 1 in steps of 0.01. The threshold is
# chosen on training folds and tested on the held-out fold
# (cross-validation), so the reported score is not tuned on its own test data.
GRID_STEP = 0.01
GRID_FOLDS = 5
RANDOM_SEED = 42

# Keyword baseline (exact keyword matching, as in Marjanovic et al., 2024):
# "any" retrieves a response if it contains at least one query word,
# "all" only if it contains every query word.
KEYWORD_MATCH_MODE = "any"

# Small words ignored by the keyword baseline (they would match everything)
STOPWORDS = {
    "a", "an", "and", "the", "of", "to", "in", "on", "for", "at", "by",
    "with", "is", "are", "was", "be", "or", "no", "not", "my", "our", "from",
}

# Settings chosen by Stage 3 that the web app will use
RETRIEVAL_SETTINGS_FILE = PROCESSED_DIR / "retrieval_settings.json"

# ---------------------------------------------------------------------------
# Stage 4: summarization
# ---------------------------------------------------------------------------
# FLAN-T5-large (~3 GB download, about 8 s per summary on a laptop CPU).
# It was chosen over FLAN-T5-base after a comparison on the synthetic test
# data: base mostly copied sentences or the "Suggested improvement:" label,
# while large wrote real summaries. Both sometimes add invented background
# (e.g. a wrong country), so summaries must be checked in the evaluation.
# To use the original T5 instead, set SUMMARY_MODEL = "t5-base" and
# SUMMARY_PROMPT = "summarize: " (T5 only understands that fixed prefix).
SUMMARY_MODEL = "google/flan-t5-large"
# Of the prompts tried, asking for the problems first and then what people
# want done gave summaries that covered both parts most often.
SUMMARY_PROMPT = ("Read these citizen complaints. Summarize what the problems "
                  "are, then what people want the government to do.\n")
# Used only when the responses do not fit in one input and are summarized
# in stages (first each group, then the group summaries together).
SUMMARY_COMBINE_PROMPT = ("Combine these summaries of citizen complaints into one "
                          "summary of the problems and what people want the "
                          "government to do.\n")

# Names a summary may mention even if no submission does, because they are
# the general context of every submission (used by src/factcheck.py).
CONTEXT_NAMES = ["Sri Lanka", "Sri Lankan", "Sri Lankans", "Government", "the government"]

# T5 reads at most 512 tokens (config.T5_TOKEN_LIMIT). At most this many of
# the retrieved responses (the most similar ones) are summarized.
SUMMARY_MAX_RESPONSES = 10

# Length and decoding of the summary. Beam search with no repeated 3-word
# phrases gives more fluent, less repetitive text than greedy decoding.
# A low minimum length is used because forcing longer summaries (30 tokens
# was tried) made the model pad them with invented details.
SUMMARY_MIN_NEW_TOKENS = 10
SUMMARY_MAX_NEW_TOKENS = 150
SUMMARY_NUM_BEAMS = 4
SUMMARY_NO_REPEAT_NGRAM = 3

# Test queries for Stage 4 and the evaluation: one per issue area. Each is
# retrieved with the Stage 3 method and its responses are summarized.
# Independent reference summaries are later written for the same responses.
EVAL_QUERIES = [
    "hospital waiting times and medicine shortages",
    "teacher shortage and school facilities",
    "road conditions and public transport",
    "water supply and power cuts",
    "court delays and legal help",
    "cost of living and welfare payments",
    "pollution and waste dumping",
    "government websites and online services",
]

# ---------------------------------------------------------------------------
# Stage 5: generation with Llama 3 (hosted API)
# ---------------------------------------------------------------------------
# Cloudflare Workers AI hosts Llama 3.3 70B Instruct for free (a daily
# allowance of 10,000 "neurons") behind an OpenAI-style API. Groq was tried
# first, but the student's Groq account offered no Llama 3 model, and
# OpenRouter only offers it paid.
# {account_id} is filled in from CLOUDFLARE_ACCOUNT_ID in .env; the token
# is LLAMA_API_KEY in .env.
# To use a paid provider instead, e.g. OpenRouter:
#   LLAMA_API_BASE_URL = "https://openrouter.ai/api/v1"
#   LLAMA_MODEL = "meta-llama/llama-3.3-70b-instruct"
LLAMA_API_BASE_URL = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"
LLAMA_MODEL = "@cf/meta/llama-3.3-70b-instruct-fp8-fast"
ENV_FILE = ROOT / ".env"

# Temperature controls randomness. Generation uses a moderate value so the
# text sounds natural; summarization uses a low value to stay close to the
# submissions. A fixed seed makes repeated runs as similar as the API allows.
GENERATION_TEMPERATURE = 0.7
LLAMA_SUMMARY_TEMPERATURE = 0.2
LLAMA_SEED = 42
GENERATION_MAX_TOKENS = 400
LLAMA_SUMMARY_MAX_TOKENS = 250

# Few-shot prompts include this many retrieved submissions as examples
FEW_SHOT_EXAMPLES = 3

# Length asked for in the generation prompt (words). Keep it close to the
# typical length of the real submissions (see "Word count" in
# preprocessing_report.txt), so the hold-out comparison is fair.
GENERATION_LENGTH_WORDS = (30, 80)

# The free tier allows a limited number of requests and tokens per minute.
# When a limit is hit, the script waits and tries again up to this many times.
LLAMA_MAX_RETRIES = 6
LLAMA_TIMEOUT_SECONDS = 60

# Hold-out run: generate from each respondent's keywords. None = everyone.
# Set a small number (e.g. 5) for a quick, cheap test run.
GENERATION_LIMIT = None

# ---------------------------------------------------------------------------
# Web app: submitting to the consultation office (prototype)
# ---------------------------------------------------------------------------
# Submissions made in the app are saved here. This is a PROTOTYPE submission
# channel: it is not connected to the Information Center's real systems.
# These records are kept apart from the study data and are never added to
# the retrieval database automatically.
SUBMISSIONS_DIR = ROOT / "data" / "submissions"
SUBMISSIONS_FILE = SUBMISSIONS_DIR / "submissions.csv"

# Issue areas offered in the submission form (same as the questionnaire)
ISSUE_AREAS = [
    "Healthcare",
    "Education",
    "Infrastructure / Transport",
    "Public Utilities (electricity, water, sanitation)",
    "Law and Justice",
    "Economic / Financial Services",
    "Environment",
    "Digital Government Services",
    "Other",
]

# ---------------------------------------------------------------------------
# Stage 7: evaluation
# ---------------------------------------------------------------------------
# BERTScore compares texts by meaning using a language model (roberta-large,
# ~1.4 GB download on first run). Rescaling with the published baseline
# spreads the scores out so differences are easier to see; raw BERTScore is
# about 0.8-0.9 for almost any pair of English sentences.
BERTSCORE_LANG = "en"
BERTSCORE_RESCALE = True

# Reference summaries written by people who did not see the model summaries.
# Copy data/processed/reference_summaries_template.csv here and fill it in
# (data/processed/ is rewritten whenever Stage 4 runs). One row per query and
# writer: the same query may appear once for each writer.
REFERENCE_SUMMARIES_FILE = RAW_DIR / "reference_summaries.csv"

# Human ratings (readability, informativeness, flow, 1-5; Li et al., 2020).
# Each rater fills in a copy of the rating sheet and saves it in this folder.
HUMAN_RATINGS_DIR = RAW_DIR / "human_ratings"
RATING_SHEET_GENERATIONS = 10      # respondents whose two generations are rated
RATING_CRITERIA = ["readability", "informativeness", "flow"]
HUMAN_RATING_TARGET = 3.5          # success criterion: mean rating >= 3.5

# System Usability Scale (Brooke, 1996): one row per participant, q1..q10
SUS_FILE = RAW_DIR / "sus_responses.csv"
SUS_BENCHMARK = 68                 # success criterion: mean SUS > 68 (Sauro, 2011)
