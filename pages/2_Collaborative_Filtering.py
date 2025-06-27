import streamlit as st
import pandas as pd
from collaborative_filtering import train_and_save_svd
from surprise import Dataset, Reader


@st.dialog("Please enter you username")
def login():
    username = st.text_input("Username")
    if st.button("Submit"):
        st.session_state.username = username
        st.rerun()


def update_svd(reviews):
    data = Dataset.load_from_df(reviews[['author_id', 'product_id', 'rating']],
                                Reader())
    trainset = data.build_full_trainset()
    train_and_save_svd(trainset)


def add_review(products, product_name, rating):
    filtered_products = products[products['product_name'] == product_name]
    product_id = filtered_products.iloc[0]['product_id']

    new_review = {
        'author_id': st.session_state.username,
        'product_id': product_id,
        'product_name': product_name,
        'rating': rating
    }

    reviews = pd.read_csv("data/filtered_reviews_processed.csv")
    reviews.loc[len(reviews)] = new_review
    reviews.to_csv("data/filtered_reviews_processed.csv")

    reviews_count = (reviews['author_id'] == st.session_state.username).sum()
    if reviews_count >= 5:
        update_svd(reviews)

    st.rerun()


st.image(
    "https://static.vecteezy.com/system/resources/previews/021/724/456/non_2x/large-group-of-people-of-different-nationality-ethnicity-and-age-isolated-on-white-background-children-adults-and-teenagers-stand-together-illustration-vector.jpg",
    use_container_width=True)

if 'username' not in st.session_state:
    login()
else:
    st.title(f"Hi, {st.session_state.username}! 👋")
    st.divider()

    st.subheader("Rate a product")
    products = pd.read_csv("data/product_info.csv")
    col1, col2 = st.columns(2)

    with col1:
        product_name = st.selectbox("Product",
                                    products['product_name'].unique())

    with col2:
        rating = st.slider("Rating", min_value=1, max_value=5, value=3, step=1)

    if st.button("Submit", use_container_width=True):
        add_review(products, product_name, rating)
