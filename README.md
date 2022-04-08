# ⚖ Weighting Schemes Report
Extend the scikit-learn classification report with custom weighting schemes.

## 🧰 Usage
```bash
from weightings_schemes_report import classification_report
classification_report(
    y_true,
    y_pred,
    average_funcs=('entropy', 'dodrans'),
    *args,
    **kwargs)
```

Package for the paper "Why only Micro-F1? Class Weighting of Measures for Relation Classification".
We provide an extended classification report that gives more F1 scores than only micro, macro and weighted.
Custom additional weighting schemes can be added to `average_funcs` by defining in `custom_scoring` and giving the string of the function name to `average_funcs`.
A function added to `custom_scoring` should take an 1d numpy array of class counts and a numpy array of labels and return (not necessarily normalized) class_weights.

Functionality of `classification_report` should otherwise be identical to the `classification_report` from scikit-learn.

## 📚 Citation

If you find the code helpful, please consider citing the following paper:
```
@inproceedings{harbecke2022why,
    title={Why only Micro-F1? Class Weighting of Measures for Relation Classification},
    author={David Harbecke and Yuxuan Chen and Leonhard Hennig and Christoph Alt},
    year={2022},
    booktitle={Proceedings of NLP Power! The First Workshop on Efficient Benchmarking in NLP}
}
```
