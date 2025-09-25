# SmartCosmetics

SmartCosmetics is a recommendation system for cosmetics products. It combines **content-based filtering** and **collaborative filtering** to provide personalized product suggestions. The system is built in Python with Jupyter notebooks for data exploration and model evaluation.

## Features

-  **Data analysis** of cosmetics products and their ingredients  
-  **Content-based recommendations** using product information and composition  
-  **Collaborative filtering** based on user ratings and reviews  
-  **Hybrid approaches** combining both strategies
-  Tool for finding similar products based on ingredients
-  Interactive Jupyter notebooks for exploration and evaluation  

## Project Structure

```
SmartCosmetics/
├── data/                        # Datasets (products, reviews, etc.)
├── models/                      # Trained models and artifacts
├── pages/                       # Streamlit pages (UI)
├── Home.py                      # Entry point for Streamlit app
├── collaborative_filtering.py   # Collaborative filtering implementation
├── content_based.py             # Content-based recommendation system
├── content_single_product.py    # Single-product recommendations
├── data_analysis.ipynb          # Data exploration & analysis
├── product_info_preprocess.ipynb# Preprocessing notebook
├── requirements.txt             # Dependencies
└── .gitignore
```

## Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/ccvetanska/SmartCosmetics.git
cd SmartCosmetics
pip install -r requirements.txt
```

## Usage

### Run notebooks
Explore the data and preprocessing:
```bash
jupyter notebook data_analysis.ipynb
```

### Run Streamlit app
```bash
streamlit run Home.py
```

This will launch the SmartCosmetics app in your browser.

## Requirements

See [requirements.txt](requirements.txt).  
Key libraries include:
- `pandas`
- `scikit-learn`
- `numpy`
- `streamlit` for the UI
- `scikit-surprise` for SVD 
- `matplotlib`, `seaborn` for visualization

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is released under the MIT License.
