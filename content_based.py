import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from sklearn.metrics import root_mean_squared_error

RATING_THRESHOLD = 3.5
TOP_K = 5

class ContentBasedModel:
    def __init__(self, products_csv="data/product_info_processed.csv"):
        products = pd.read_csv(products_csv)
        products = products.dropna(subset=['ingredients']).reset_index(drop=True)

        self.products_df = products
        self.tfidf = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf.fit_transform(products['ingredients'])
        self.product_id_to_index = dict(zip(products['product_id'], products.index))
        self.index_to_product_id = dict(zip(products.index, products['product_id']))
        self.id_to_name = dict(zip(products['product_id'], products['product_name']))

    def recommend(self, reviews_df, user_id, top_k=TOP_K):
        user_reviews = reviews_df[reviews_df['author_id'] == user_id]
        liked = user_reviews[user_reviews['rating'] >= RATING_THRESHOLD]

        if liked.empty:
            return None

        liked_indices = [self.product_id_to_index[pid] for pid in liked['product_id']
                         if pid in self.product_id_to_index]
        if not liked_indices:
            return None

        liked_vectors = self.tfidf_matrix[liked_indices]
        user_profile = np.asarray(liked_vectors.mean(axis=0))

        similarities = cosine_similarity(user_profile, self.tfidf_matrix).flatten()

        rated_products = set(user_reviews['product_id'])
        rec_indices = np.argsort(similarities)[::-1]

        recs = [self.index_to_product_id[idx] for idx in rec_indices
                if self.index_to_product_id[idx] not in rated_products][:top_k]

        result = pd.DataFrame({
            'Product Name': [self.id_to_name[pid] for pid in recs],
            'Score': [round(similarities[self.product_id_to_index[pid]], 4) for pid in recs]
        })

        result.index = result.index + 1
        return result

    def evaluate(self, test_df, train_df, k=TOP_K, sample_size=None, random_state=42, max_products=None):
        if max_products is not None:
            self.tfidf_matrix = self.tfidf_matrix[:max_products]
            self.products_df = self.products_df.iloc[:max_products]
            self.product_id_to_index = dict(zip(self.products_df['product_id'], self.products_df.index))
            self.index_to_product_id = dict(zip(self.products_df.index, self.products_df['product_id']))
            self.id_to_name = dict(zip(self.products_df['product_id'], self.products_df['product_name']))

        user_liked = train_df.groupby('author_id')['product_id'].apply(list)
        grouped_test = test_df.groupby('author_id')

        user_ids = list(user_liked.keys())
        if sample_size is not None:
            np.random.seed(random_state)
            user_ids = np.random.choice(user_ids, size=min(sample_size, len(user_ids)), replace=False)

        all_users = []
        user_profiles = []
        test_data = []

        for user in user_ids:
            if user not in grouped_test or user not in user_liked:
                continue

            liked_products = user_liked[user]
            liked_indices = [self.product_id_to_index[pid] for pid in liked_products if pid in self.product_id_to_index]
            if not liked_indices:
                continue

            liked_vectors = self.tfidf_matrix[liked_indices]
            user_profile = np.asarray(liked_vectors.mean(axis=0))

            all_users.append(user)
            user_profiles.append(user_profile)
            test_row = grouped_test.get_group(user).iloc[0]
            test_data.append((user, test_row['product_id'], test_row['rating']))

        if not user_profiles:
            return 0.0, 0.0, 0.0

        user_profiles_matrix = np.vstack(user_profiles)
        similarity_matrix = cosine_similarity(user_profiles_matrix, self.tfidf_matrix)

        y_true = []
        y_pred = []
        correct = 0
        total_precision = 0

        for i, (user, actual_pid, actual_rating) in enumerate(test_data):
            if actual_pid not in self.product_id_to_index:
                continue

            similarities = similarity_matrix[i]
            rated_products = set(train_df[train_df['author_id'] == user]['product_id'])

            rec_indices = np.argsort(similarities)[::-1]
            recs = [self.index_to_product_id[idx] for idx in rec_indices
                    if self.index_to_product_id[idx] not in rated_products][:k]

            pid_index = self.product_id_to_index[actual_pid]
            pred_score = similarities[pid_index]

            y_true.append(actual_rating)
            y_pred.append(pred_score)

            if actual_pid in recs:
                correct += 1
            total_precision += int(actual_pid in recs) / k

        count = len(test_data)
        precision = total_precision / count if count else 0
        recall = correct / count if count else 0
        rmse = root_mean_squared_error(y_true, y_pred, squared=False) if y_true else 0

        return round(precision, 4), round(recall, 4), round(rmse, 4)

    
def split_reviews_userwise(reviews_df, min_liked=2, rating_threshold=3.5):
    train_rows = []
    test_rows = []

    grouped = reviews_df.groupby('author_id')

    for user_id, group in grouped:
        liked = group[group['rating'] >= rating_threshold]
        if len(liked) < min_liked:
            continue  

        liked_shuffled = liked.sample(frac=1, random_state=42)
        test_rows.append(liked_shuffled.iloc[0]) 
        train_rows.extend(liked_shuffled.iloc[1:].to_dict(orient='records')) 

    train_df = pd.DataFrame(train_rows)
    test_df = pd.DataFrame(test_rows)

    return train_df, test_df

if __name__ == '__main__':
    cb_model = ContentBasedModel()
    reviews = pd.read_csv("data/filtered_reviews_processed.csv", dtype={'author_id': str})
    train_df, test_df = split_reviews_userwise(reviews)
    # precision, recall, rmse = cb_model.evaluate(test_df, train_df, k=5, max_products=500)
    # print(f"Precision@5: {precision}")
    # print(f"Recall@5: {recall}")
    # print(f"RMSE: {rmse}")
    
    cb_model = ContentBasedModel()
    recs = cb_model.recommend(reviews, user_id="6152272933")
    print(recs)