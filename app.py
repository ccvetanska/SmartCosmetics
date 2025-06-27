import streamlit as st
from content_single_product import prepare_data_single_product, recommend_products_similar_to
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

df, feature_names, indices, tfidf_matrix = prepare_data_single_product('data/product_info_cosmetics.csv')
context = {
    "df": df,
    "feature_names": feature_names,
    "indices": indices,
    "tfidf_matrix": tfidf_matrix
}

st.title("SmartCosmetics Recommender")

tab1, tab2, tab3, tab4 = st.tabs(["Similar by ingredients", "Recommend for user by content", "Collaborative filtering", "Hybrid"])

# ------------------ TAB 1 ------------------
with tab1:
    st.write("Personalized recommendations based on a selected product's ingredients.")

    product_name = st.selectbox("Choose product:", df['product_name'].unique())

    product_ingredients = df.loc[df['product_name'] == product_name, 'ingredients'].values[0]
    ingredient_list = product_ingredients.split(', ')

    st.subheader("Ingredients of the selected product:")
    st.code(product_ingredients, language='markdown', height=200, wrap_lines=True)

    selected_ingredients = st.multiselect("Which of the ingredients are important for you?", ingredient_list)

    ingredient_weights = {}
    for ingr in selected_ingredients:
        weight = st.slider(f"Weight for '{ingr}'", 1.0, 5.0, 2.0, 0.1)
        ingredient_weights[ingr] = weight ** 2

    if st.button("See similar"):
        results = recommend_products_similar_to(
            product_name=product_name,
            top_n=10,
            ingredient_weights=ingredient_weights,
            context=context
        )
        st.subheader("Recommended:")

        base_ingredients = set(ingredient_list)
        weighted_ingredients = set(ingredient_weights.keys())

        idx = context['indices'].get(product_name)
        if ingredient_weights:
            weight_vector = np.ones(len(context["feature_names"]))
            for ingr, weight in ingredient_weights.items():
                if ingr in context["feature_names"]:
                    index = list(context["feature_names"]).index(ingr)
                    weight_vector[index] = weight

            mod_matrix = context["tfidf_matrix"].multiply(weight_vector)
            query_vector = context["tfidf_matrix"][idx].multiply(weight_vector)
            cosine_scores = cosine_similarity(query_vector, mod_matrix).flatten()
        else:
            cosine_scores = cosine_similarity(context["tfidf_matrix"][idx], context["tfidf_matrix"]).flatten()

        for i, row in results.iterrows():
            product_idx = row.name
            similarity = cosine_scores[product_idx]
            st.markdown(f"**{row['product_name']}** – _Similarity: {similarity:.2%}_")

            recommended_ingredients = row['ingredients'].split(', ')
            highlighted = []
            for ingr in recommended_ingredients:
                is_base = ingr in base_ingredients
                is_weighted = ingr in weighted_ingredients

                if is_weighted:
                    html = f"<span style='background-color:#fff3b0'>{ingr}</span>"
                elif is_base:
                    html = f"<span style='background-color:#d1ffd1'>{ingr}</span>"
                else:
                    html = ingr

                highlighted.append(html)

            html_ingredients = ", ".join(highlighted)
            st.markdown(html_ingredients, unsafe_allow_html=True)

# ------------------ TAB 2 ------------------
with tab2:
    st.write("In progress")
    
# ------------------ TAB 3 ------------------
with tab3:
    st.write("In progress")
    
# ------------------ TAB 4 ------------------
with tab4:
    st.write("In progress")