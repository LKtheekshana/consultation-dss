# Consultation DSS

LLM-based Decision Support System for automating citizen consultation
processes, with special reference to the Information Center, Sri Lanka.

## Project structure

```
consultation-dss/
├── app.py                    Stage 6: Streamlit web app (entry point)
├── app_pages/
│   ├── citizen.py            citizen portal: search, summary, draft & submit
│   └── office.py             office inbox for submitted items
├── config.py                 settings shared by all stages
├── requirements.txt          Python libraries
├── data/
│   ├── raw/                  input data (form export goes here)
│   └── processed/            outputs of each stage (created automatically)
├── src/
│   ├── preprocess.py         Stage 1: cleaning and sentence splitting
│   ├── embed.py              Stage 2: sentence embeddings (all-mpnet-base-v2)
│   ├── retrieve.py           Stage 3: retrieval and threshold comparison
│   ├── summarize.py          Stage 4: summarization (FLAN-T5)
│   ├── factcheck.py          flags names in summaries that are not in the sources
│   ├── submissions.py        saving submissions, reference numbers, receipts
│   ├── webapp.py             cached models and data for the web app
│   ├── llm.py                API client for Llama 3 (Cloudflare Workers AI)
│   ├── generate.py           Stage 5: generation + Llama summaries
│   └── evaluate.py           Stage 7: evaluation
└── tests/
    ├── test_preprocess.py    automatic checks for Stage 1
    ├── test_embed.py         Stage 2 (uses a small stand-in model, works offline)
    ├── test_retrieve.py      Stage 3
    ├── test_summarize.py     Stage 4 (stand-in model, works offline)
    ├── test_generate.py      Stage 5 (fake API, works offline, no key needed)
    ├── test_submissions.py   submitting from the web app
    ├── test_factcheck.py     invented-name check
    └── test_evaluate.py      Stage 7 (stand-in for BERTScore, works offline)
```



## First-time setup (Windows, in the VS Code terminal)

```
cd path\to\consultation-dss
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux, use `source .venv/bin/activate` instead of the second line.

## Stage 1: Preprocessing

```
python src/preprocess.py
```

This reads the file set in `config.INPUT_FILE` and writes to `data/processed/`:

| File | Contents |
|---|---|
| `clean_responses.csv` | one row per usable response, with the combined `response_text` used by later stages |
| `sentences.csv` | each response split into sentences (input for embeddings) |
| `removed_responses.csv` | dropped rows and the reason (empty, duplicate, too short) |
| `preprocessing_report.txt` | counts and word-count statistics |

To run the automatic checks: `pytest -q`

## Stage 2: Embeddings

```
python src/embed.py
```

The first run downloads all-mpnet-base-v2 (about 420 MB). Each sentence is
embedded, the sentence vectors of a response are averaged (mean pooling),
and the result is scaled to length 1. Responses marked `needs_translation`
are skipped.

| File | Contents |
|---|---|
| `embeddings.pkl` | response ids and one 768-number vector per response |
| `embedding_check.txt` | checks: vector lengths, similarity within vs between issue areas |

If the model works, "Same issue area" similarity in `embedding_check.txt`
should be clearly higher than "Different issue area".

## Stage 3: Retrieval and thresholds

```
python src/retrieve.py
```

Compares four ways of choosing which responses to return, plus a baseline:

| Method | Rule | Setting in `config.py` |
|---|---|---|
| static | similarity >= fixed T | `STATIC_THRESHOLD` |
| dynamic | similarity >= mean + k·std of the query's scores | `DYNAMIC_K` |
| grid | T searched from 0 to 1 in steps of 0.01 (cross-validated) | `GRID_STEP`, `GRID_FOLDS` |
| top-k | the k most similar | `TOP_K` |
| keyword (baseline) | exact keyword matching (Marjanovic et al., 2024) | `KEYWORD_MATCH_MODE` |

Each respondent's own keywords are used as a query, with their own response
left out. Another response counts as relevant if it has the same
`issue_area` (main label) or shares at least one ticked issue in
`issues_selected` (stricter label). Precision, recall and F1 are averaged
over queries. The grid-search threshold is chosen on 4/5 of the queries and
tested on the other 1/5, so it is never scored on the queries it was tuned on.

| File | Contents |
|---|---|
| `retrieval_report.txt` | results table (P, R, F1, mean retrieved) and Wilcoxon test vs baseline |
| `retrieval_results.csv` | scores for every query, method and label |
| `grid_search_curve.csv` | P/R/F1 at every threshold |
| `retrieval_settings.json` | method and threshold the web app will use |

## Input data: the questionnaire export

The pipeline reads the real questionnaire export, `data/raw/form_responses.csv`
(set in `config.INPUT_FILE`). To create it:

1. In Google Forms, open the **Responses** tab, click the three-dot menu and
   choose **Download responses (.csv)**.
2. Unzip the download and save the CSV as `data/raw/form_responses.csv`.
   Do not open and re-save it in Excel (this can break Sinhala and Tamil
   text), and do not delete or rename columns.
3. If you edited any question wording in the form, update the matching
   title in `config.FORM_COLUMN_MAP` so the columns are recognized.
4. Run `python src/preprocess.py` and check the report.

The synthetic test file used while building the pipeline
(`SYNTHETIC_test_dataset.csv`) and all outputs made from it were deleted on 9 October 2026.
It is no longer read by the system, and no results from it are reported.
The automated tests do not use it.

## Sinhala and Tamil responses

The language models work in English, so Sinhala and Tamil responses are
translated before Stage 2. Translations live in `data/raw/translations.csv`
(one row per response, matched by the form's Timestamp):

| Column | Meaning |
|---|---|
| `problem_original`, `suggestion_original`, `keywords_original` | the respondent's own words (copied from the export) |
| `problem_en`, `suggestion_en`, `keywords_en` | the English translation |
| `translated_by` | who made the translation (the current drafts: "Claude (draft)") |
| `checked_by` | **empty until a person has checked the translation against the original** |
| `check_notes` | what was corrected, if anything |

**Checking the translations.** Sinhala rows are checked by the researcher,
Tamil rows by a Tamil speaker. Correct the English text where needed, then
put your name in `checked_by`. Edit the file in VS Code, or import it into
Google Sheets (File → Import → Upload) and download it again as CSV. Do
**not** open and save it in Excel, which can break Sinhala and Tamil text.

Stage 1 uses a translation only when `checked_by` is filled in
(`config.TRANSLATIONS_REQUIRE_CHECK = True`). Unchecked responses stay in
`clean_responses.csv` with `needs_translation = True` and are skipped by
Stage 2. After checking, run `python src/preprocess.py` again. Record how many
responses were translated, by whom the drafts were made, and who checked them.

Responses written in Sinhala or Tamil using English letters ("Singlish")
are not detected automatically; list their Timestamps in
`config.FORCE_LANGUAGE`.

## Removals decided by the researcher

Responses removed after reading them (for example off-topic answers or
reworded copies) are listed in `config.MANUAL_EXCLUSIONS` with the reason,
keyed by Timestamp. They appear in `removed_responses.csv` and the
preprocessing report as "researcher decision: ...", next to the automatic
removals (empty, duplicate, too short).

## Privacy

`.gitignore` stops real participant data and your `.env` API key from being
uploaded to GitHub. Phone numbers, emails and NIC numbers found in free text
are replaced with placeholders during cleaning.

## Stage 4: Summarization

```
python src/summarize.py
```

The first run downloads FLAN-T5-large (about 3 GB); each summary then takes
about 8 seconds on a laptop CPU. For every test query in
`config.EVAL_QUERIES` (one per issue area) the script retrieves responses
with the Stage 3 method and summarizes them.

T5 reads at most 512 tokens, so at most `SUMMARY_MAX_RESPONSES` (10)
responses are used, the most similar first. If they still do not fit, they
are summarized in groups and the group summaries are then combined.

| File | Contents |
|---|---|
| `summaries.csv` | query, response ids used, tokens, stages, time and the summary |
| `summarization_report.txt` | the same as a readable table, plus every summary |
| `reference_summaries_template.csv` | the responses each summary was made from, with an empty `reference_summary` column |

**Reference summaries for the evaluation:** give
`reference_summaries_template.csv` to people who have **not** seen the
model's summaries. They read the responses in each row and write their own
short summary in `reference_summary` (and their name or initials in
`written_by`). Stage 7 compares the model's summaries with these using
ROUGE and BERTScore. Do this with the real data, not the synthetic file.

**Known limitation:** FLAN-T5 sometimes adds background that is not in the
responses (for example the wrong country). `src/factcheck.py` finds names
(organizations, places, acronyms) in a summary that appear in none of its
source submissions; the web app shows a warning for them, and Stage 7 can
count how often each summarizer does this. It only checks capitalized
names, not every claim. This was seen in testing, and
the human ratings also measure it.

## Stage 5: Generation (Llama 3.3 70B Instruct via Cloudflare Workers AI)

**One-time setup (free):**

1. Create a free Cloudflare account at https://dash.cloudflare.com/sign-up.
2. Create an API token with the **Workers AI** template at
   https://dash.cloudflare.com/profile/api-tokens.
3. Copy your **Account ID** (shown on the Workers AI page of the dashboard).
4. Copy `.env.example` to `.env` and fill in `LLAMA_API_KEY` (the token) and
   `CLOUDFLARE_ACCOUNT_ID`, **in `.env`, not in `.env.example`**
   (`.env.example` is meant to be shared; `.env` is in `.gitignore`, so it
   is never uploaded).

The free plan has a daily allowance (10,000 "neurons"). If it runs out,
the script stops with a message; run it again the next day (the allowance
resets at 00:00 UTC, 5:30 am in Sri Lanka) and it continues where it
stopped.

```
python src/generate.py
```

1. **Hold-out generation.** For every respondent, a submission is generated
   from *their* keywords twice: **zero-shot** (keywords only) and
   **few-shot** (plus the 3 most similar *other* submissions as examples;
   the respondent's own response is never shown to the model). Stage 7
   compares these with what the respondent actually wrote.
2. **Llama summaries.** The exact responses FLAN-T5 summarized in Stage 4
   are summarized again with Llama, so both summarizers can be compared.

Only cleaned text is sent to the API (phone, email and NIC numbers were
replaced in Stage 1). Generated text is cleaned the same way as a safety
net, and every output is marked `ai_generated = True`.

If the API is briefly busy, the script waits and retries automatically. If it stops, run it again: finished generations are kept.
For a quick test, set `GENERATION_LIMIT = 5` in `config.py`.

| File | Contents |
|---|---|
| `generations.csv` | keywords, prompt type, example ids, generated problem/suggestion, real text, tokens, time |
| `summaries_llama.csv` | Llama summary for each Stage 4 test query |
| `generation_report.txt` | counts, format compliance, lengths, tokens, example output, FLAN-T5 vs Llama summaries |

**Ethics:** real (anonymized) responses are sent to Cloudflare, a
third-party service. Make sure your participant information sheet and ethics approval
cover this before running Stage 5 on the real data.

## Stage 7: Evaluation

```
python src/evaluate.py
```

The first run downloads roberta-large for BERTScore (about 1.4 GB). The
script can be run again at any time; parts that need people show "Pending"
until their files exist.

| Part | What is measured | Needs |
|---|---|---|
| A. Generation | ROUGE-1, ROUGE-L, BLEU, BERTScore of each generated text against the respondent's real submission; zero-shot vs few-shot (Wilcoxon); control row against an unrelated submission | nothing |
| B. Summaries | the same metrics for FLAN-T5 and Llama against reference summaries | `data/raw/reference_summaries.csv` |
| C. Invented names, response times | per summarizer and function | nothing |
| D. Human ratings | readability, informativeness, flow (1-5) per system; Krippendorff's alpha | filled rating sheets |
| E. Usability | SUS score (0-100); how much users edited AI drafts | `data/raw/sus_responses.csv`, app submissions |
| F. Success criteria | met / not met / pending | the above |

**What people need to do:**

1. **Reference summaries.** Copy `data/processed/reference_summaries_template.csv`
   to `data/raw/reference_summaries.csv` (Stage 4 rewrites the one in
   `processed/`). Two people who have not seen the model summaries each
   write a 40-80 word summary of the problems and suggestions in each row,
   without AI tools. Put each writer's summaries in their own rows (copy the
   row, keep `query` the same) and their name in `written_by`.
2. **Human ratings.** `data/raw/human_ratings/rating_sheet_TEMPLATE.csv`
   lists the outputs in random order without saying which system wrote
   them. Each rater saves a copy as `rating_<name>.csv` in the same folder
   and fills in `readability`, `informativeness` and `flow` from 1 (very
   poor) to 5 (very good). `rating_key.csv` links items to systems: do not
   show it to raters.
3. **SUS.** Participants use the web app (search, read a summary, prepare
   and submit a submission), then answer `data/raw/sus_form.txt`. Enter
   each participant's answers in `data/raw/sus_responses.csv` (copy
   `sus_responses_TEMPLATE.csv`), one row per participant, values 1-5.

Outputs: `evaluation_report.txt` (all evaluation tables), `generation_scores.csv`,
`summary_scores.csv`.

## Web app

```
streamlit run app.py
```

Opens at http://localhost:8501 (needs Stages 1-3 to have been run). It has
two pages, chosen at the top:

**Citizen portal.** Enter 2-3 keywords and press **Search**:

- **Related submissions:** the retrieved responses with their similarity scores.
  If even the best match is below `WEAK_MATCH_SIMILARITY` (0.37), a warning
  suggests more specific keywords.
- **Summary:** press **Summarize** for a Llama summary, labelled as
  AI-generated. If Llama cannot be reached (no key, no internet, daily
  limit), FLAN-T5 runs locally as a backup and a notice says so.
- **Draft & submit:** write a submission yourself, or press **Generate AI
  draft** (few-shot, needs the `.env` key) and edit the draft.
  Choose the issue area, tick the box confirming you have checked the text,
  and press **Submit to consultation office**. You get a reference number
  (e.g. `SUB-20261009-0001`) and can **download a copy** as a `.txt` file;
  a copy of an unsent draft can be downloaded too. Submissions containing a
  phone number, email address or NIC number are blocked until the details
  are removed.

**Office inbox.** The submissions received: counts, share of AI-assisted
submissions, how much citizens changed the AI drafts, a filter by issue
area, the full text of a selected submission (with the original AI draft),
and a CSV download.

The app uses only the configuration that performed best in the evaluation:
the retrieval method saved by Stage 3 (`retrieval_settings.json`, dynamic
threshold), Llama summaries and few-shot drafts. The other thresholds, FLAN-T5
summaries and zero-shot drafts remain in the pipeline scripts, so the
evaluation can be rerun. The usability (SUS) sessions on 15 October 2026 used
the earlier version, which still offered these options. Stop the app with
Ctrl+C in the terminal.

**Submissions are a prototype channel.** They are saved in
`data/submissions/submissions.csv`, not sent to the Information Center's
real systems, and they are kept apart from the study data (they are never
added to the retrieval database). Each record keeps the original AI draft
next to the final text, with `edited` and `edit_ratio` (0 = sent as
drafted, 1 = completely rewritten, word level), which can be reported in
the evaluation. In a real deployment the office inbox would need a staff login.
`data/submissions/` is in `.gitignore`.

## Troubleshooting

**`OSError: [WinError 4551] An Application Control policy has blocked this file ... torch\lib\shm.dll`**

Windows 11 Smart App Control blocks PyTorch's unsigned DLLs. Turn it off in
**Windows Security → App & browser control → Smart App Control settings → Off**,
then open a new terminal and run the stage again. Microsoft Defender
antivirus stays on. Note: on many Windows builds Smart App Control cannot
be switched back on without resetting Windows.
