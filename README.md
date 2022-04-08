# Weighting schemes report

Usage:
```bash
from weightings_schemes_report import classification_report
classification_report(
    y_true,
    y_pred,
    average_funcs=('entropy', 'dodrans'),
    *args,
    **kwargs)
```

Additional weighting schemes can be added to `average_funcs` by defining in `custom_scoring` and giving the string of the function name to `average_funcs`.
A function added to `custom_scoring` should take an 1d numpy array and return one.

Functionality of `classification_report` should otherwise be identical to the `classification_report` from scikit-learn.
