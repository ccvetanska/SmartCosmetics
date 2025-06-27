import pandas as pd
from surprise import Dataset, Reader, SVD, SVDpp
from surprise.model_selection import train_test_split
from surprise import accuracy
from surprise.dump import dump
from collections import defaultdict


def prepare_data():
    reviews = pd.read_csv("data/filtered_reviews.csv",
                          dtype={'author_id': 'str'})

    reviews = reviews[~reviews['author_id'].astype(str).str.
                      startswith("orderGen")].reset_index(drop=True)

    data = Dataset.load_from_df(reviews[['author_id', 'product_id', 'rating']],
                                Reader())

    return train_test_split(data, test_size=0.2)


def precision_recall_at_k(predictions, k, threshold):
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return sum(prec for prec in precisions.values()) / len(precisions), sum(
        rec for rec in recalls.values()) / len(recalls)


def train_and_save_svd(trainset):
    model = SVD()
    model.fit(trainset)
    dump("models/svd.pkl", algo=model)

    return model


def eval_svd(testset, algo, k, threshold):
    predictions = algo.test(testset)

    rmse = accuracy.rmse(predictions, verbose=False)
    precision, recall = precision_recall_at_k(predictions, k, threshold)

    print(f'Test RMSE: {round(rmse, 4)}')
    print(
        f'Test Precision@{k} with threshold {threshold}: {round(precision, 4)}'
    )
    print(f'Test Recall@{k} with threshold {threshold}: {round(recall, 4)}')


if __name__ == "__main__":
    trainset, testset = prepare_data()
    model = train_and_save_svd(trainset)
    eval_svd(testset, model, 5, 3.5)
