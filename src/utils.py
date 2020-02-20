from typing import Iterable, List, Tuple

import implicit
import numpy as np
import pandas as pd
from scipy import sparse


def transform_to_item_user_csr_matrix(purchases: pd.DataFrame) -> sparse.csr_matrix:
    item_users = sparse.coo_matrix(
        (
            np.ones(purchases.customer_id.size, dtype=np.float32),
            (purchases.product_id, purchases.customer_id,),
        )
    ).tocsr()
    return item_users


def get_recommendations(
    model: implicit.als.AlternatingLeastSquares,
    user_ids: Iterable[int],
    item_users: sparse.csr_matrix,
) -> List[List[int]]:
    user_items = item_users.T.tocsr()
    recommendations = []
    for user_id in user_ids:
        recommendations.append(
            [x[0] for x in model.recommend(userid=user_id, user_items=user_items, N=10)]
        )
    return recommendations


def get_purchases_by_customer(
    purchases: pd.DataFrame,
) -> Tuple[List[int], List[List[int]]]:
    relevant = (
        purchases.groupby("customer_id")["product_id"]
        .apply(lambda s: s.values.tolist())
        .reset_index()
    )
    relevant.rename(columns={"product_id": "product_ids"}, inplace=True)
    return relevant["customer_id"].tolist(), relevant["product_ids"].tolist()
