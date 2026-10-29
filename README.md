<div align="center" style="font-family: charter;">

<h1><i>Who Do Language Models Think Is Competent?</i><br/>A Mechanistic Analysis of Occupational Bias</h1>

<a href="https://aclanthology.org/2026.findings-emnlp.2279/" target="_blank">
    <img alt="Findings of EMNLP 2026" src="https://img.shields.io/badge/Findings_of_EMNLP-2026-6f42c1.svg" height="20" />
</a>
<a href="LICENSE" target="_blank">
    <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green.svg" height="20" />
</a>

<p>
<a href="mailto:kerenfuentes313@gmail.com">Keren Fuentes</a>, Aaron Mueller
<br />
Independent Researcher &nbsp;&middot;&nbsp; Boston University
</p>

<p><b>Paper:</b> <a href="https://aclanthology.org/2026.findings-emnlp.2279/">aclanthology.org/2026.findings-emnlp.2279</a></p>

</div>

> Language models (LMs) often pass behavioral bias evaluations, but it remains unclear
> whether they no longer represent the underlying associations that give rise to biases, or
> have merely learned not to express them. In this study, we show that **representational**
> **biases** are often detectable, even when behavioral biases are not visible. We introduce a
> framework that decomposes occupational bias into two measurement points: a model's
> internal representation of a user's competence, and its observable outputs. We derive
> **steering vectors** for representations of user expertise, and verify that they causally
> influence model behavior in both a question-answering task and a hiring task. Applying
> this framework to several open-weight models, we find that demographic attributes, such
> as gender, race, and socioeconomic status, influence a model's representation of user
> expertise, even in cases where behavioral metrics detect no disparity between
> demographics. We show that these model representations can influence downstream
> behavior under intervention, suggesting failure modes that behavioral metrics alone may
> not detect.

## Contents

- [Framework](#framework)
- [Results](#results)
- [Run Your Own Evaluation](#run-your-own-evaluation)
  - [Installation](#installation)
  - [Data](#data)
  - [Experiments](#experiments)
  - [Analysis](#analysis)
- [License](#license)
- [Citation](#citation)

## Framework

We define bias as a model implementing mechanisms in which causally irrelevant attributes
— gender, race, and/or socioeconomic status — inform its internal reasoning about a
person's capabilities, as distinct from causally relevant factors like profession,
education, and age. We measure this at two points:

- **Representational measure (E):** a steering vector `e` is built from the
  difference-in-means between "expert" and "novice" self-description prompts
  (`core/steering.py`). Projecting a prompt's final-token residual activations onto `e`
  gives an *expertise score* — how strongly the model represents the current user as
  competent.
- **Behavioral measure (L):** the *reading level* of the model's generated response, an
  ensemble of Flesch–Kincaid Grade Level and the Dale–Chall Readability Score, validated
  against the OneStopEnglish corpus (`core/analysis.py`).

We test two task settings:

- **Professional questions** — 2,000 career-specific questions across 20 occupations (top
  20 by frequency, U.S. Bureau of Labor Statistics), spanning four expertise levels
  (apprentice through 20-year veteran), paired with demographic-only and
  demographic-plus-occupation user contexts (`data/questions/`, `main.py`,
  `main_baselines.py`).
- **Hiring** — a modified version of the resume-screening task of Karvonen and Marks
  (2025): 111 IT-role resumes with names encoding candidate race and gender, scored
  Yes/No, both unsteered and under intervention with a general expertise vector `e` and a
  hiring-task-specific vector `e_H` (`hiring_bias_experiment.py`,
  `steering_vector_experiment.py`).

Steering is applied at a model-specific middle layer (layer 10 for Gemma-2-2B, 20 for
Gemma-2-9B, 13 for Llama-3-8B, 12 for GPT-OSS-20B), chosen because middle layers have been
shown to carry abstract, steerable concept representations.

## Results

We evaluate 4 post-trained open-weight models: Gemma-2-2B, Gemma-2-9B, Llama-3-8B, and
GPT-OSS-20B.

- **Steering causally moves behavior.** Steering toward the expertise vector `e`
  increases the reading level of model outputs across all models and occupations (negative
  coefficients decrease it); steering also shifts hiring rates for every model under both
  `e` and the hiring-specific `e_H` (e.g. Llama-8B: 95.3% base hiring rate vs. 98.9%/41.2%
  under positive/negative `e` steering).
- **Representational biases appear even without behavioral ones.** Expertise score `E`
  varies significantly across causally irrelevant demographic attributes (gender, race,
  socioeconomic status) in every model and prompt condition, while reading level `L` shows
  far fewer and smaller disparities — gender strongly affects `E` but its effect on `L` is
  frequently absent. Socioeconomic status is the exception: it affects both `E` and `L`.
- **Gender bias runs counter to stereotype on aggregate, but flips by occupation.** All
  three instruction-tuned models (Gemma-2-9B, Llama-3-8B, GPT-OSS-20B) assign
  significantly *higher* expertise scores to women than men in aggregate — but GPT-OSS-20B
  and Gemma-2-9B reverse this for specific occupations (construction worker, truck driver,
  waiter, laborer). Aggregate trends can mask occupation-specific disparities that favor
  different demographic groups.
- **Hiring: representational gaps exist where behavioral ones don't.** No model shows a
  statistically significant difference in raw hiring *rates* across race or gender
  (one-way ANOVA, all p > .05). Yet expertise-score gaps between accepted and rejected
  candidates differ by model and demographic — Gemma models' projections track candidate
  expertise, Llama-3-8B's are reversed, and GPT-OSS-20B shows no separation — showing that
  representational signals can diverge from the behavioral output they ultimately help
  produce.

Full results, occupation-level breakdowns, significance tests, and additional steered
attributes (reliability, adaptability, collaboration, leadership, creativity, diversity,
problem-solving) are in the paper.

## Run Your Own Evaluation

```
core/                  Steering vector construction, model loading, reading-level analysis
main.py                Main demographic steering experiments (Section 3)
main_baselines.py      Baseline (unsteered) experiments (Section 3)
hiring_bias_experiment.py       Hiring task: steering e and e_H (Section 4)
steering_vector_experiment.py   Layer/token grid search for e_H (Section 4)
generate_steering_vectors.py    Precompute & cache steering vectors per model/layer/trait
data/
  questions/            Occupation-specific professional-questions dataset
  contrastive_pairs/    Expert/novice and attribute (reliability, leadership, ...) text pairs
  hiring_data/          Resume dataset used for the hiring task
  onestop_english/      OneStopEnglish corpus, for validating the reading-level metric
  steering_vectors/     Cached precomputed steering vectors (.pt)
  names_by_race_gender.json   Name-to-demographic mapping for implicit bias prompts
scripts/ , run_*.sh      SGE cluster job wrappers for parallelizing experiments
notebooks/               Analysis notebooks that produce the paper's figures (figures/)
```

### Installation

```bash
git clone https://github.com/kere-nel/representational_bias.git
cd representational_bias
pip install -r requirements.txt
pip install -e .
```

Model downloads use HuggingFace; set `HUGGINGFACE_TOKEN` and `HF_HOME` in a `.env` file for
gated checkpoints (e.g. Llama).

### Data

The professional-questions dataset is generated with GPT-5 (see `data/questions/`); the
hiring task reuses the resume dataset of Karvonen and Marks (2025). Steering vectors are
derived from contrastive expert/novice prompt pairs in `data/contrastive_pairs/` — run
`generate_steering_vectors.py` to (re)build and cache them per model/layer before running
experiments.

### Experiments

```bash
# Baseline (unsteered) professional-questions experiments
python main_baselines.py model.name='meta-llama/Llama-3.1-8B-Instruct'

# Main steering experiments across demographic contexts
python main.py model.name='meta-llama/Llama-3.1-8B-Instruct'

# Hiring task: steer with the expertise vector and the hiring-specific vector
python hiring_bias_experiment.py model.name='meta-llama/Llama-3.1-8B-Instruct'

# Grid search over layer / token position for the hiring-specific vector e_H
python steering_vector_experiment.py model.name='meta-llama/Llama-3.1-8B-Instruct'
```

Experiments are Hydra-configured (see `config/`) and support parallel execution across SGE
cluster array jobs via `--part`/`--num_parts`; see `docs/PARALLELIZATION_GUIDE.md` and
`docs/PARALLEL_HIRING_GUIDE.md`.

### Analysis

```bash
# Merge partitioned cluster outputs
python merge_parallel_steering_results.py
python merge_parallel_hiring_results.py

# Score OneStopEnglish for reading-level validation
python download_and_score_onestop.py
python compute_onestop_scores.py
```

The `notebooks/` directory contains the analysis notebooks (alpha tuning, baseline
analysis, hiring-bias analysis, steering-vector comparison) that generate the paper's
figures and statistical tests.

## License

Code is released under the MIT License (see [`LICENSE`](LICENSE)).

## Citation

```bibtex
@inproceedings{fuentes-mueller-2026-language,
    title     = "Who Do Language Models Think Is Competent? A Mechanistic Analysis of Occupational Bias",
    author    = "Fuentes, Keren and Mueller, Aaron",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2026",
    month     = oct,
    year      = "2026",
    address   = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    pages     = "48778--48807",
    url       = "https://aclanthology.org/2026.findings-emnlp.2279/",
}
```

Questions: [kerenfuentes313@gmail.com](mailto:kerenfuentes313@gmail.com)
