import streamlit as st
import pandas as pd
from collaborative_filtering import train_and_save_svd
from surprise import Dataset, Reader
from surprise.dump import load
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

    reviews_count = (reviews['author_id'] == st.session_state.username).sum()
    if reviews_count >= 5:
        data = Dataset.load_from_df(
            reviews[['author_id', 'product_id', 'rating']], Reader())
        trainset = data.build_full_trainset()
        train_and_save_svd(trainset)

    st.rerun()


def get_content_recommendations(reviews):
    user_reviews = reviews[reviews['author_id'] == st.session_state.username]
    if len(user_reviews) < 5:
        return None

    recs = cb_model.recommend(reviews, st.session_state.username)
    if recs is None or recs.empty:
        return None
    recs = recs.rename(columns={'Score': 'Similarity Rating'})
    return recs


def get_cf_recommendations(reviews):
    user_reviews = reviews[reviews['author_id'] == st.session_state.username]
    if len(user_reviews) < 5:
        return None

    rated_products = user_reviews['product_id'].unique()
    all_products = reviews['product_id'].unique()
    unseen = [p for p in all_products if p not in rated_products]

    _, model = load('models/svd.pkl')

    preds = [(p, model.predict(st.session_state.username, p).est)
             for p in unseen]
    top_preds = sorted(preds, key=lambda x: x[1], reverse=True)[:5]

    product_ids = [pid for pid, _ in top_preds]
    estimated_ratings = [float(rating) for _, rating in top_preds]

    id_to_name = reviews.set_index('product_id')['product_name'].to_dict()
    product_names = [
        id_to_name[pid] for pid in product_ids if pid in id_to_name
    ]

    df = pd.DataFrame({
        'Product Name': product_names,
        'Estimated Rating': estimated_ratings
    })

    df.index = df.index + 1

    return df


st.image(
    "https://img.freepik.com/free-vector/emotional-feedback-concept-illustration_114360-21832.jpg?semt=ais_hybrid&w=740",
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

    tab1, tab2 = st.tabs(["Content-based", "Collaborative filtering"])

    with tab1:
        c_recommendations = get_content_recommendations(reviews)
        if c_recommendations is None:
            st.info(
                'Please rate at least 5 products to get personalised recommendations',
                icon="ℹ️")
        else:
            st.table(c_recommendations)
    with tab2:
        cf_recommendations = get_cf_recommendations(reviews)
        if cf_recommendations is None:
            st.info(
                'Please rate at least 5 products to get personalised recommendations',
                icon="ℹ️")
        else:
            st.table(cf_recommendations)

    st.divider()
    st.subheader("Your reviews")

    display_df = user_reviews[['product_name',
                               'rating']].rename(columns={
                                   'product_name': 'Product Name',
                                   'rating': 'Rating'
                               })

    st.table(display_df)
