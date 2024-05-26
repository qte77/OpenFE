# OpenFE: An efficient automated feature generation tool

Forked from [IIS-Li-Group/OpenFE](https://github.com/IIIS-Li-Group/OpenFE), see also [original documentation](https://openfe-document.readthedocs.io/en/latest/).

## Simplyfied usage

```python
pip install OpenFE@git+https://github.com/qte77/OpenFE -qq
ofe = OpenFE()
features = ofe.fit(data=train_x, label=train_y, train_index=index_col, **ofep)
ofe.new_features_list
train_x, test_x = ofe.transform(train_x, test, features, n_jobs=n_jobs)
# only for testing after OpenFE is done, custom get_score() necessary
score = get_score(train_x, test, train_y, test_y)
```

## Description of data operations

Feature generation methods used ordered by categorial and numerical. Creates features and uses `lightgbm.LGBMRegressor` and `lightgbm.LGBMClassifier` to rank them according to importance. 

* All: Freq
* Numerical: Abs, Log, Sqrt, Square, Sigmoid, Round, Residual
* Num2Num: Add, Substract, Multiply, Divise, Max, Min
* Cat2Num: GroupByThenMin, GroupByThenMax, GroupByThenMean, GroupByThenMedian, GroupByThenStd, GroupByThenRank
* Cat2Cat: Combine, CombineThenFreq, GroupByThenNUnique
* Symmetry: Add, Subsctract, Multiply, Divise, Min, Max, Combine, CombineThenFreq

### Example [CombineThenFreq](https://github.com/qte77/OpenFE/blob/c99c96c544a0f620ffe8781753ca9342355bb0bd/openfe/FeatureGenerator.py#L120)

```python
elif self.name == "CombineThenFreq":
    temp = d1.astype(str) + '_' + d2.astype(str)
    temp[d1.isna() | d2.isna()] = np.nan
    value_counts = temp.value_counts()
    value_counts.loc[np.nan] = np.nan
    new_data = temp.apply(lambda x: value_counts.loc[x])
```

### Example [GroupByThenRank](https://github.com/qte77/OpenFE/blob/c99c96c544a0f620ffe8781753ca9342355bb0bd/openfe/FeatureGenerator.py#L103)

- [pandas.DataFrame.groupby](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html)
- [Pandas Group by: split-apply-combine](https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#groupby)
- [Pandas Cookbook Grouping](https://pandas.pydata.org/pandas-docs/stable/user_guide/cookbook.html#cookbook-grouping)

```python
elif self.name == 'GroupByThenRank':
    new_data = d1.groupby(d2).rank(ascending=True, pct=True)
```

## Core program flow

```
OpenFE
|-fit(data, label, metric)
| |- get_init_score() -> init_metric
| |- stage1_select() -> return_results
| \- stage2_select() -> results
\-transform(X_train, X_test, new_features_list) -> _train, _test
```

## Core structure

```
root
|- examples
|- openfe
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

[ ] Multi-process with `concurrent.futures.ProcessPoolExecutor` not working for `OPenFE._evaluate()`
[ ] Merge redundant methods like `OpenFE._calculate_and_evaluate()` and `OpenFE._calculate_and_evaluate_multiprocess()`
[ ] Make sure `random.shuffle()` does not intere with the transformed data, i.e. index and values are properly output
[ ] Try Mojo for concurrency
