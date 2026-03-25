# ResearchLab - Evaluating Cross-Country University Policies on the Usage of LLMs in Higher Education.

This project analyzes university LLM policy documents across five countries using automated extraction, text preprocessing, topic modeling, semantic framing analysis, and multi-method sentiment and stance detection.

## Required Initial Files

These files must exist before running the pipeline:

| File                       | Location |
| -------------------------- | -------- |
| urls.txt                   | data/    |
| university_list.csv        | manual/  |
| sentiment_manual.xlsx      | manual/  |
| annotations.csv            | manual/  |
| raw_genai_data.csv         | data/    |
| manual_corrected_data.xlsx | data/    |

## Setup

```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows

# 2. Install dependencies
pip install -r requirements.txt

#3 
pip install -e .

# 4. Download NLP models and resources
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('vader_lexicon')"
```

**Python version:** 3.12 (Windows)

**Config system:** All scripts read paths from config/config.yaml via from config import CONFIG.

## Execution Pipeline

Run all commands from the project root directory.

### Data Collection & Preprocessing

| Step | Script                          | Command                                                                                 |
| ---- | ------------------------------- | --------------------------------------------------------------------------------------- |
| 0    | Manual Step                     | Prepare data/urls.txt and `manual/university_list.csv`                              |
| 1    | `extraction/extract_data.py`  | `python -m extraction.extract_data`                                                   |
| 2    | Manual Step                     | Correct extraction errors → save `raw_genai_data.csv`+`manual_corrected_data.xlsx` |
| 3    | `manual/merge_manual_data.py` | `python -m manual.merge_manual_data`                                                  |
| 4    | `cleaning/clean_data.py`      | `python -m cleaning.clean_data`                                                       |

### Topic Modeling

| Step | Script                                       | Command                                             |
| ---- | -------------------------------------------- | --------------------------------------------------- |
| 5    | `topic_modeling/train_lda_model.py`        | `python -m topic_modeling.train_lda_model`        |
| 6    | `topic_modeling/lda_uni_topic_modeling.py` | `python -m topic_modeling.lda_uni_topic_modeling` |
| 7    | `topic_modeling/bert_topic_modeling.py`    | `python -m topic_modeling.bert_topic_modeling`    |
| 8    | `topic_modeling/bert_topic_labeling.py`    | `python -m topic_modeling.bert_topic_labeling`    |

### Topic Profiling & Distribution

| Step | Script                                     | Command                                           |
| ---- | ------------------------------------------ | ------------------------------------------------- |
| 9.1  | `analysis/uni_multi_topic_profiling.py`  | `python -m analysis.uni_multi_topic_profiling`  |
| 9.2  | `analysis/country_topic_profiles.py`     | `python -m analysis.country_topic_profiles`     |
| 10.1 | `analysis/country_distribution.py`       | `python -m analysis.country_distribution`       |
| 10.2 | `analysis/country_topic_analysis_all.py` | `python -m analysis.country_topic_analysis_all` |
| 10.3 | `analysis/topic_cooccurence.py`          | `python -m analysis.topic_cooccurence`          |

### Framing & Semantic Analysis

| Step | Script                                       | Command                                             |
| ---- | -------------------------------------------- | --------------------------------------------------- |
| 11.1 | `framing_analysis/framing.py`              | `python -m framing_analysis.framing`              |
| 11.2 | `framing_analysis/framing_interpret.py`    | `python -m framing_analysis.framing_interpret`    |
| 11.3 | `analysis/affective_language_analysis.py`  | `python -m analysis.affective_language_analysis`  |
| 11.4 | `analysis/institutional_roles_analysis.py` | `python -m analysis.institutional_roles_analysis` |
| 11.5 | `analysis/curriculum_framing_analysis.py`  | `python -m analysis.curriculum_framing_analysis`  |
| 11.6 | `analysis/country_framing_summary.py`      | `python -m analysis.country_framing_summary`      |
| 11.7 | `analysis/word_cloud.py`                   | `python -m analysis.word_cloud`                   |

### Model Evaluation

| Step | Script                         | Command                               |
| ---- | ------------------------------ | ------------------------------------- |
| 12   | `analysis/run_evaluation.py` | `python -m analysis.run_evaluation` |

### Sentiment, Stance & Tone

| Step | Script                                               | Command                                                     |
| ---- | ---------------------------------------------------- | ----------------------------------------------------------- |
| 13.1 | `sentiment_stance/policy_tone_distilbert.py`       | `python -m sentiment_stance.policy_tone_distilbert`       |
| 13.2 | `sentiment_stance/stance_detection.py`             | `python -m sentiment_stance.stance_detection`             |
| 13.3 | `sentiment_stance/policy_stance_analysis.py`       | `python -m sentiment_stance.policy_stance_analysis`       |
| 13.4 | `sentiment_stance/policy_sentiment_analysis.py`    | `python -m sentiment_stance.policy_sentiment_analysis`    |
| 13.5 | `sentiment_stance/sentiment_stance_analysis.py`    | `python -m sentiment_stance.sentiment_stance_analysis`    |
| 13.6 | `sentiment_stance/roberta_sentiment.py`            | `python -m sentiment_stance.roberta_sentiment`            |
| 14.1 | Manual Step                                          | Annotate `manual/sentiment_manual.xlsx`                   |
| 14.2 | `sentiment_stance/manual_tone_frequency.py`        | `python -m sentiment_stance.manual_tone_frequency`        |
| 14.3 | `sentiment_stance/manual_tone_groups.py`           | `python -m sentiment_stance.manual_tone_groups`           |
| 15   | `sentiment_stance/manual_annotation_evaluation.py` | `python -m sentiment_stance.manual_annotation_evaluation` |

## Outputs

* `data/` — stores all intermediate and final datasets (cleaned text, topic assignments, stance scores)
* `models/` — stores trained LDA model (models/lda/) and BERTopic model (models/bertopic_model/)
* `analysis_results/ `— stores all charts, tables, reports, and evaluation outputs organized by analysis type

## Project Structure

```ResearchLab/
├── config/
│   ├── __init__.py
│   └── config.yaml
├── extraction/
│   └── extract_data.py
├── manual/
│   ├── merge_manual_data.py
│   ├── university_list.csv
│   ├── sentiment_manual.xlsx
│   └── annotations.csv
├── cleaning/
│   └── clean_data.py
├── topic_modeling/
│   ├── train_lda_model.py
│   ├── lda_uni_topic_modeling.py
│   ├── bert_topic_modeling.py
│   └── bert_topic_labeling.py
├── analysis/
│   ├── uni_multi_topic_profiling.py
│   ├── country_topic_profiles.py
│   ├── country_distribution.py
│   ├── country_topic_analysis_all.py
│   ├── topic_cooccurence.py
│   ├── affective_language_analysis.py
│   ├── institutional_roles_analysis.py
│   ├── curriculum_framing_analysis.py
│   ├── country_framing_summary.py
│   ├── word_cloud.py
│   ├── evaluation_metrics.py
│   └── run_evaluation.py
├── framing_analysis/
│   ├── framing.py
│   └── framing_interpret.py
├── sentiment_stance/
│   ├── policy_tone_distilbert.py
│   ├── stance_detection.py
│   ├── policy_stance_analysis.py
│   ├── policy_sentiment_analysis.py
│   ├── sentiment_stance_analysis.py
│   ├── roberta_sentiment.py
│   ├── manual_tone_frequency.py
│   ├── manual_tone_groups.py
│   └── manual_annotation_evaluation.py
├── data/
├── models/
├── analysis_results/
│   ├── country/
│   ├── framing/
│   ├── multi_topic_profiling/
│   ├── rq1/ + figures/
│   ├── rq3/ + figures/
│   ├── sentiment_analysis/ + figures/
│   ├── Stance_detection/ + figures/
│   ├── manual_sentiment/
│   └── topic_cooccurrence/
├── requirements.txt
└── README.md
```

## Help

* If you get `ModuleNotFoundError: No module named 'config'`, ensure the `.pth` file is set up correctly

```
echo %CD% > .venv\Lib\site-packages\researchlab.pth
```

* If config changes are not reflected after editing `config.yaml`, clear Python cache

```
Get-ChildItem -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
```
