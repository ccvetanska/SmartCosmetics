import streamlit as st
import pandas as pd
from collaborative_filtering import train_and_save_svd
from surprise import Dataset, Reader
from surprise.dump import load


@st.dialog("Please enter you username")
def login():
    username = st.text_input("Username")
    if st.button("Submit"):
        st.session_state.username = username
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


def get_recommendations(reviews):
    user_reviews = reviews[reviews['author_id'] == st.session_state.username]
    if len(user_reviews) < 5:
        return None

    rated_products = user_reviews['product_id'].tolist()
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
    "https://static.vecteezy.com/system/resources/previews/021/724/456/non_2x/large-group-of-people-of-different-nationality-ethnicity-and-age-isolated-on-white-background-children-adults-and-teenagers-stand-together-illustration-vector.jpg",
    use_container_width=True)

if 'username' not in st.session_state:
    login()
else:
    st.title(f"Hi, {st.session_state.username}! 👋")
    st.divider()

    reviews = pd.read_csv("data/filtered_reviews_processed.csv")

    st.subheader("Rate a product")
    col1, col2 = st.columns(2)

    with col1:
        product_name = st.selectbox("Product",
                                    reviews['product_name'].unique())

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

    user_reviews = reviews[reviews['author_id'] ==
                           st.session_state.username].reset_index(drop=True)
    user_reviews.index = user_reviews.index + 1

    st.table(user_reviews[['product_name', 'rating']])
