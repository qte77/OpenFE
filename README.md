# OpenFE: An efficient automated feature generation tool

Forked from [IIS-Li-Group/OpenFE](https://github.com/IIIS-Li-Group/OpenFE)

## Usage

```python
ofe = OpenFE()
features = ofe.fit(data=train_x, label=train_y, train_index=index_col, **ofep)
ofe.new_features_list
train_x, test_x = ofe.transform(train_x, test, features, n_jobs=n_jobs)
# only for testing after OpenFE is done, custom get_score() necessary
score = get_score(train_x, test, train_y, test_y)
```

## Core program flow

* TODO

## Core structure

```
root
|- examples
\- openfe
|   |- __init__.py
|   |- FeatureGenerator.py
|   |- FeatureSelector.py
|   |- openfe.py
|   \- utils.py
|- README.md
\- setup.py
```

## Changed

* Added `n_estimators` to `OpenFE` to communicate with `LGBM`
* Added `sklearn.metrics.r2_score`
* Added more verbosity levels
* Added code folding for structure

## TODO

* Multi-process with `concurrent.futures.ProcessPoolExecutor` not working for `OPenFE._evaluate()`
* Merge redundant methods like `OpenFE._calculate_and_evaluate()` and `OpenFE._calculate_and_evaluate_multiprocess()`
* Try Mojo for concurrency
