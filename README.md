# BHASHA: Achieving Sarcasm Interpreted Translation

## Project Overview
**BHASHA** focuses on improving machine translation systems by enabling them to interpret and accurately translate **sarcastic** English tweets into **honest Telugu** translations.  
We propose two pipelines:
- **Pipeline A**: English (sarcastic) → English (honest) → Telugu (honest)
- **Pipeline B**: English (sarcastic) → Telugu (honest) (direct)

Breaking the task into interpretation first and translation second significantly boosts translation quality, especially for low-resource languages like Telugu.

---

## Key Contributions
- Fine-tuned transformer models (**T5**, **BART**, **mT5**, **mBART**) for sarcasm interpretation and translation.
- Created a **high-quality manually corrected** English-to-Telugu sarcasm interpretation dataset.
- Comprehensive evaluation using **BLEU**, **ROUGE**, **PINC** scores, and **human evaluation** for **adequacy** and **fluency**.

---

## Methods
- **Sarcasm Interpretation**:  
  Fine-tuning seq2seq models (T5, BART) to reinterpret sarcastic English into honest English.
  
- **Translation**:  
  Translating the honest English into honest Telugu using pre-trained multilingual translation models (mT5, mBART).

- **Dataset**:  
  Built by extending and manually correcting the **Sarcasm SIGN dataset**.

- **Evaluation**:  
  - Automatic: BLEU, ROUGE, PINC scores.
  - Human: Adequacy and fluency ratings by Telugu language experts.

---

## Results

### English-to-English Sarcasm Interpretation

| **Model** | **BLEU** | **ROUGE-1** | **ROUGE-2** | **ROUGE-L** | **PINC** |
|:---------:|:--------:|:-----------:|:-----------:|:-----------:|:--------:|
| SIGN (baseline) | 66.96 | 70.34 | 42.81 | 69.98 | 47.11 |
| T5-base | 84.34 | 87.89 | 80.90 | 87.37 | 15.97 |
| T5-large | 85.29 | 89.28 | 82.83 | 88.95 | 13.83 |
| BART-large | **86.32** | 86.40 | 80.73 | 86.21 | 11.06 |

---

### English-to-Telugu Sarcasm Translation

| **Pipeline** | **Interpretation Model** | **Translation Model** | **BLEU** | **ROUGE-1** | **ROUGE-2** | **ROUGE-L** |
|:------------:|:-------------------------:|:----------------------:|:--------:|:-----------:|:-----------:|:-----------:|
| A | T5-base | mBART-large-50-many-to-many-mmt | 35.39 | 15.17 | 6.04 | 14.85 |
| A | T5-large | mBART-large-50-many-to-many-mmt | **35.80** | 15.00 | 6.26 | 14.73 |
| A | BART-large | mBART-large-50-many-to-many-mmt | 35.69 | 15.34 | 6.69 | 15.04 |
| A | T5-base | mBART-large-50-one-to-many-mmt | 33.92 | 15.44 | 6.63 | 15.34 |
| A | T5-large | mBART-large-50-one-to-many-mmt | 33.12 | 15.46 | 6.62 | 15.30 |
| A | BART-large | mBART-large-50-one-to-many-mmt | 33.47 | **15.68** | **6.81** | **15.59** |
| A | T5-base | mT5-base | 17.64 | 11.00 | 3.73 | 10.91 |
| A | T5-large | mT5-base | 17.37 | 12.18 | 4.47 | 12.05 |
| A | BART-large | mT5-base | 17.01 | 11.84 | 4.02 | 11.61 |
| B | - | mBART-large-50-many-to-many-mmt | 31.69 | 14.48 | **7.07** | 14.42 |
| B | - | mBART-large-50-one-to-many-mmt | 30.39 | 14.72 | 6.97 | 14.60 |
| B | - | mT5-base | 13.54 | 8.88 | 2.95 | 8.86 |

---

### Human Evaluation

| **Metric**   | **Pipeline A** | **Pipeline B** |
|:------------:|:--------------:|:--------------:|
| Adequacy     | 3.8 / 4         | 3.2 / 4         |
| Fluency      | 3.88 / 4        | 3.04 / 4        |

> **Note:** Pipeline A (Interpretation + Translation) consistently outperforms Pipeline B (Direct Translation) across both automatic and human evaluation metrics.

---

## Future Work
- Enable direct translation of sarcasm while preserving sarcastic intent.
- Expand the dataset to include sarcastic expressions directly translated into Telugu and other languages.
- Incorporate **context recognition** to improve sarcasm interpretation even further.

---

## Acknowledgements
Thanks to the University of Florida and all contributors who assisted with data annotation and model evaluation.

---
