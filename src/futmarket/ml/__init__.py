"""Machine-learning layer: features, datasets, and the never-ending training loop.

Models are histogram gradient boosting via scikit-learn (same family as LightGBM,
no system OpenMP needed) and sit behind a thin interface so the estimator can be
swapped without touching the feature or training code.
"""
