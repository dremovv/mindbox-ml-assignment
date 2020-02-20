from os import environ

import implicit


def get_baseline_model() -> implicit.als.AlternatingLeastSquares:
    # disable internal multithreading to speed up implicit.als.AlternatingLeastSquares.fit()
    environ["MKL_NUM_THREADS"] = "1"
    environ["OPENBLAS_NUM_THREADS"] = "1"

    # we iterated through hyper parameters and measured map@10 score on test set
    # the parameters below provide 80-th percentile of score
    # we intentionally do not use parameters with best test score
    model = implicit.als.AlternatingLeastSquares(
        factors=20, iterations=7, regularization=100.0
    )
    return model
