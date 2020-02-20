from os import environ

import implicit


def get_baseline_model() -> implicit.als.AlternatingLeastSquares:
    # disable internal multithreading to speed up implicit.als.AlternatingLeastSquares.fit()
    environ["MKL_NUM_THREADS"] = "1"
    environ["OPENBLAS_NUM_THREADS"] = "1"

    # to get these hyper parameters, we used random search
    # in (factors, iterations, regularization) space
    # then we took parameters that provide 80-percentile of map@10 score on test_purchases
    # it means you can iterate through parameters and get better score

    model = implicit.als.AlternatingLeastSquares(
        factors=20, iterations=7, regularization=100.0
    )
    return model
