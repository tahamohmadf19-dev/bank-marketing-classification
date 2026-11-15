# Bank Marketing Classification

Predicts whether a bank client will subscribe to a term deposit, using Apache Spark MLlib on the UCI Bank
Marketing dataset (45,211 records, 11.7% positive).

**Read this first:** a model that always answers "no" scores 88.3% accuracy on this data, so the notebook's
90.5% accuracy says little alone. The strongest feature, `duration`, is the length of the call, which is only
known after the call ends. A model meant to decide whom to call cannot use it. Both effects are measured below.

**Status:** learning project. It is not deployed and has not been tested on a real campaign.

## Results

### Spark notebook

Rerun from a clean environment with PySpark 3.5.3 and Java 17. Outputs are saved in the notebook. The split is

`requirements.txt` now pins PySpark 3.5.8, the patch release in the same 3.5 line that fixes PYSEC-2026-3966. The saved results above are from the 3.5.3 run and were not re-run on 3.5.8.
a random 60/20/20 from `randomSplit` (not stratified). F1 here is `MulticlassClassificationEvaluator`'s
**weighted F1**, which is dominated by the majority class. All models include `duration`.

| Model | Validation ROC-AUC | Validation accuracy | Validation weighted F1 |
|---|---:|---:|---:|
| Decision Tree | 0.6684 | 0.8317 | 0.8555 |
| Random Forest | 0.9178 | 0.8367 | 0.8610 |
| Linear SVC | 0.9039 | 0.8986 | 0.8723 |

A manual grid over `numTrees` (50, 100, 200) and `maxDepth` (5, 10, 15) chose 100 trees and depth 15 by
validation F1. This is a loop on one validation split, not `CrossValidator`.

| Best Random Forest, test split | Value |
|---|---:|
| ROC-AUC | 0.9268 |
| Accuracy | 0.9051 |
| Weighted F1 | 0.8919 |

Earlier saved outputs showed 0.9221, 0.8988 and 0.8831. The figures changed with the rerun, so small differences
between these versions should not be read as improvements.

### Baselines and the `duration` feature

[analysis/baselines.py](analysis/baselines.py) runs a separate scikit-learn analysis with a stratified 80/20
split, seed 42. Results are in [analysis/baseline_results.json](analysis/baseline_results.json). Its split differs
from the notebook's, so the two tables do not compare exactly.

| Model | Accuracy | ROC-AUC | PR-AUC | Yes-class recall | Yes-class F1 |
|---|---:|---:|---:|---:|---:|
| Always "no" | 0.8830 | 0.5000 | 0.1170 | 0.0000 | 0.0000 |
| Logistic regression, with `duration` | 0.9012 | 0.9056 | 0.5451 | 0.3478 | 0.4518 |
| Logistic regression, without `duration` | 0.8933 | 0.7717 | 0.4111 | 0.1786 | 0.2815 |
| Random Forest, with `duration` | 0.9047 | 0.9288 | 0.6201 | 0.3214 | 0.4410 |
| Random Forest, without `duration` | 0.8953 | 0.8029 | 0.4528 | 0.1994 | 0.3083 |

**How to read this:** the rows without `duration` are the honest estimate for planning calls. Even there, the
default threshold finds only about one in five subscribers.

## Setup

Requires Python 3.11 and Java 17. Spark 3.5 does not run on newer Python versions without extra care.

```bash
git clone https://github.com/tahamohmadf19-dev/bank-marketing-classification.git
cd bank-marketing-classification
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download `bank-full.csv` from the [UCI repository](https://archive.ics.uci.edu/dataset/222/bank+marketing) and
place it at `data/bank-full.csv`. To use another location, set `BANK_DATA_PATH`.

```bash
jupyter notebook bank_marketing_classification.ipynb
python analysis/baselines.py
python -m pytest
```

The full notebook takes several minutes on a laptop, mostly in the grid search. The tests are static checks of
the saved notebook and results. They do not retrain any model.

## Limitations

- The notebook's models keep `duration`, so their scores overstate what is useful before a call.
- No threshold was tuned for campaign cost, and no cross-validation was used.
- The dataset is from a Portuguese bank between 2008 and 2010. The results may not transfer elsewhere.

## Reference

Moro, S., Cortez, P., and Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing.
*Decision Support Systems*, 62, 22-31.

## License

MIT. See [LICENSE](LICENSE).
