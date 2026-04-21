# Group Work Project 1 (GWP1) - Financial Data

**Course:** WQU Course 02 - 600 Financial Data  
**Topic:** Behavioral Social Media Data: Exploratory Data Analysis & Sentiment Modeling

## Overview
This repository contains the analytical pipeline for examining alternative behavioral social media data, specifically historical Reddit comments. The goal of this project is to parse highly unstructured social sentiment and quantify its statistical relationship to the price returns of major assets (BTC-USD).

## Project Structure
* **`Behavioral_Social_Media_EDA.ipynb`**: The core Jupyter Notebook containing the data loading, rolling sentiment analysis, and initial EDA.
* **`advanced_eda_builder.py`**: A Python utility script designed to programmatically inject advanced, academic-grade Exploratory Data Analysis (EDA) and sophisticated NLP metrics into the base Jupyter Notebook.

## Advanced Analytical Phases
The expanded EDA pipeline executes the following methodologies:
1. **Structural EDA**: Analyzes comment length distributions and time-series volume density to detect anomalous bot-like engagement.
2. **Sophisticated NLP Tokenization**: Employs NLTK for stopword removal, lemmatization, and frequency distributions of community-specific jargon.
3. **Granular VADER Sentiment**: Visualizes the absolute distribution of VADER compound scores and calculates rolling sentiment volatility.
4. **Financial Correlation Metrics**: Connects sentiment first-differences against daily logarithmic BTC-USD returns, outputting mathematical Pearson and Spearman rank correlation coefficients.

## Setup & Execution
Ensure you have the following requisite libraries installed in your Python/Jupyter environment:
`pandas`, `numpy`, `matplotlib`, `yfinance`, `nltk` (with `vader_lexicon` & `stopwords`).

To safely inject the advanced statistical cells into the notebook, run the builder script:
```bash
python3 advanced_eda_builder.py
```
Then, open the updated `.ipynb` file in your preferred Jupyter environment (local or Google Colab) and execute "Run All" to render the visualizations.

## Credits
* **Avinash Sharma**: Implemented early behavioral data fetching and MLP tuning in `avinash_sharma_projectWQ.ipynb`. (WQU team member, avinashhbs@gmail.com)
