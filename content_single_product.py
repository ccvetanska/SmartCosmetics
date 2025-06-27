import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import numpy as np

def prepare_data_single_product(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=['ingredients']).reset_index(drop=True)
    df['ingredients'] = df['ingredients'].apply(lambda x: x.strip("[]").strip("'").strip('"'))
    tfidf = TfidfVectorizer(tokenizer=lambda x: x.split(', '), token_pattern=None)
    tfidf_matrix = tfidf.fit_transform(df['ingredients'])
    indices = pd.Series(df.index, index=df['product_name']).drop_duplicates()
    feature_names = tfidf.get_feature_names_out()
    return df, feature_names, indices, tfidf_matrix

def parse_ingredients(s):
    s = s.strip("[]").strip("'").strip('"')
    items = [i.strip() for i in s.split(',')]
    items = [i.rstrip('.') for i in items if i]
    return items

def compute_similarity(idx, tfidf_matrix, weight_vector, use_dot_product=True):
    query_vector = tfidf_matrix[idx].multiply(weight_vector)
    mod_matrix = tfidf_matrix.multiply(weight_vector)
    
    if use_dot_product:
        return query_vector.dot(mod_matrix.T).toarray().flatten()
    else:
        return cosine_similarity(query_vector, mod_matrix).flatten()
    
def recommend_products_similar_to(context, product_name, top_n=5, ingredient_weights=None, use_dot_product=True):
    df = context["df"]
    feature_names = context["feature_names"]
    indices = context["indices"]
    tfidf_matrix = context["tfidf_matrix"]
    
    idx = indices.get(product_name)
    if idx is None:
        return pd.DataFrame({"product_name": [], "ingredients": []})

    weight_vector = np.ones(len(feature_names))

    if ingredient_weights:
        feature_names_norm = [f.strip().lower() for f in feature_names]
        ingredient_weights_norm = {k.strip().lower(): v for k, v in ingredient_weights.items()}

        for ingr, weight in ingredient_weights_norm.items():
            if ingr in feature_names_norm:
                index = feature_names_norm.index(ingr)
                weight_vector[index] = weight

        query_vector = tfidf_matrix[idx].multiply(weight_vector)
        mod_matrix = tfidf_matrix.multiply(weight_vector)

        if use_dot_product:
            sim_scores = query_vector.dot(mod_matrix.T).toarray().flatten()
        else:
            sim_scores = cosine_similarity(query_vector, mod_matrix).flatten()
    else:
        sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
        
    applied = [(f, weight_vector[i]) for i, f in enumerate(feature_names) if weight_vector[i] != 1.0]
    print("Applied weights:", applied)
    
    sim_scores = list(enumerate(sim_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != idx][:top_n]

    product_indices = [i[0] for i in sim_scores]
    return df.iloc[product_indices][['product_name', 'ingredients']]

if __name__ == '__main__':    
    df, feature_names, indices, tfidf_matrix = prepare_data_single_product('data/product_info_cosmetics.csv')
    context = {
        "df": df,
        "feature_names": feature_names,
        "indices": indices,
        "tfidf_matrix": tfidf_matrix
    }
    print(recommend_products_similar_to(context, "African Beauty Butter Collection Deluxe Tin (54 Thrones)", 
                                        top_n=5))