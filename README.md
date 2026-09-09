# Stock Identification from Order-Book Sequences

Sequence-classification experiments for the CFM–ENS data challenge: identify a stock from a sequence of 100 order-book updates.

**Start here:** [GRU and CNN–GRU notebook](cfm-gru-benchmark.ipynb).

## Implemented approach

The notebook uses Polars to read and transform order-book events, pandas for one-hot encoding, scikit-learn for numeric standardization, and TensorFlow/Keras for classification.

Features include prices and sizes, bid/ask imbalance, event categories, and previous-event fields grouped by observation and order identifiers. Rows are reshaped into sequences of 100 events.

| Experiment | Architecture | Training configuration in notebook |
| --- | --- | --- |
| Bidirectional GRU | GRU(64), dropout 0.1, GRU(32), dropout 0.1, softmax; both GRUs are bidirectional. | Adam, categorical cross-entropy, batch size 64, up to 50 epochs, validation fraction 0.25. |
| CNN–bidirectional GRU | Conv1D(64, kernel size 3), max pooling, bidirectional GRU(64) and GRU(32), dropout 0.2, softmax. | Adam, categorical cross-entropy, batch size 64, up to 50 epochs, validation fraction 0.20. |

Both use early stopping on validation loss with patience 3 and restore the best weights.

## Data and entry point

The challenge data is not included. The notebook expects these files under `/kaggle/input/cfmens/`:

- `X_train_N1UvY30.csv`
- `y_train_or6m3Ta.csv`
- `X_test_m4HAPAP.csv`

Supply the challenge files and adapt those paths to your environment. Open `cfm-gru-benchmark.ipynb` in Jupyter or the corresponding Kaggle environment.

Imported packages include NumPy, pandas, Polars, scikit-learn, and TensorFlow. Package versions are not pinned; the notebook uses older Polars method names, so compatibility needs checking before execution.

A prediction cell writes `submission_laset.csv`. No verified leaderboard score is documented here.

## Research status and next steps

This repository preserves exploratory challenge experiments. Several details need correction before using validation accuracy as a reliable benchmark:

- The scaler is fitted before the Keras validation split; fit preprocessing on training observations only.
- Train and test dummy columns are created separately; align feature schemas explicitly.
- Confirm event ordering and label alignment before reshaping rows into sequences.
- The feature called `vwap` is calculated row by row as price times bid size divided by bid size; it is not an aggregated VWAP.
- Use a documented split that accounts for the available time/group structure, compare against simple baselines, and report repeatability across seeds.

This work concerns stock identification from supplied sequences; it does not establish a tradable return forecast.
