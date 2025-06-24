import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np

df = pd.read_csv('data/product_info.csv')
df = df.dropna(subset=['ingredients']).reset_index(drop=True)

tfidf = TfidfVectorizer(tokenizer=lambda x: x.split(', '), token_pattern=None)
tfidf_matrix = tfidf.fit_transform(df['ingredients'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(df.index, index=df['product_name']).drop_duplicates()

feature_names = tfidf.get_feature_names_out()

def recommend(product_name, top_n=5, ingredient_weights=None):
    idx = indices.get(product_name)
    if idx is None:
        return f"Product '{product_name}' not found."

    if ingredient_weights:
        weight_vector = np.ones(len(feature_names))
        for ingr, weight in ingredient_weights.items():
            if ingr in feature_names:
                index = list(feature_names).index(ingr)
                weight_vector[index] = weight
        mod_matrix = tfidf_matrix.multiply(weight_vector)
        sim_matrix = linear_kernel(mod_matrix, mod_matrix)
    else:
        sim_matrix = cosine_sim

    sim_scores = list(enumerate(sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    product_indices = [i[0] for i in sim_scores]
    return df.iloc[product_indices][['product_name', 'ingredients']]

# Example
print(recommend("Baomint Leave In Conditioning Styler", top_n=5))