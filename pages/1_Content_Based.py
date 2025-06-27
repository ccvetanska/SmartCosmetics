import streamlit as st
import pandas as pd
from content_based import ContentBasedModel

cb_model = ContentBasedModel("data/product_info_processed.csv")


@st.dialog("Please enter you username")
def login():
    username = st.text_input("Username")
    if st.button("Submit"):
        st.session_state.username = str(username)
        st.rerun()


def add_review(reviews, product_name, rating):
    filtered_products = reviews[reviews['product_name'] == product_name]
    product_id = filtered_products.iloc[0]['product_id']

    new_review = {
        'author_id': st.session_state.username,
        'product_id': product_id,
        'product_name': product_name,
        'rating': rating
    }

    reviews.loc[len(reviews)] = new_review
    reviews.to_csv("data/filtered_reviews_processed.csv", index=False)
    st.rerun()


def get_recommendations(reviews):
    user_reviews = reviews[reviews['author_id'] == st.session_state.username]
    if len(user_reviews) < 5:
        return None

    recs = cb_model.recommend(reviews, st.session_state.username)
    if recs is None or recs.empty:
        return None
    recs = recs.rename(columns={'Score': 'Similarity Rating'})
    return recs


st.image(
    "https://as1.ftcdn.net/v2/jpg/02/09/37/22/1000_F_209372242_IlSCLiys8H6TF1ePlC0EVuxS25Al4KEC.jpg",
    use_container_width=True)

if 'username' not in st.session_state:
    login()
else:
    st.title(f"Hi, {st.session_state.username}! 👋")
    st.divider()

    reviews = pd.read_csv("data/filtered_reviews_processed.csv",
                          dtype={'author_id': 'str'})
    user_reviews = (reviews[reviews['author_id'] == st.session_state.username]
                    ).reset_index(0)
    user_reviews.index = user_reviews.index + 1

    st.subheader("Rate a product")
    col1, col2 = st.columns(2)

    with col1:
        rated_products = user_reviews['product_name'].unique()
        all_products = reviews['product_name'].unique()
        unseen = [p for p in all_products if p not in rated_products]

        product_name = st.selectbox("Product", unseen)

    with col2:
        rating = st.slider("Rating", min_value=1, max_value=5, value=3, step=1)

    if st.button("Submit", use_container_width=True):
        add_review(reviews, product_name, rating)

    st.divider()
    st.subheader("Recommended for you")

    recommendations = get_recommendations(reviews)
    if recommendations is None:
        st.info(
            'Please rate at least 5 products to get personalised recommendations',
            icon="ℹ️")
    else:
        st.table(recommendations)

    st.divider()
    st.subheader("Your reviews")

    display_df = user_reviews[['product_name',
                               'rating']].rename(columns={
                                   'product_name': 'Product Name',
                                   'rating': 'Rating'
                               })

    st.table(display_df)
