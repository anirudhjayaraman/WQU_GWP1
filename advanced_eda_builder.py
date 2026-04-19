import nbformat

def add_advanced_eda(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    cells_to_add = [
        # Phase 1
        nbformat.v4.new_markdown_cell("## 5. Rigorous Data Cleaning & Structural EDA\nLet's deeply analyze the structure of the Reddit comments. We'll start by looking at comment length to identify bot noise or extreme outliers."),
        nbformat.v4.new_code_cell("""import numpy as np
# Add a length column
df['text_length'] = df['text'].apply(lambda x: len(str(x)))

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
# Histogram of comment lengths
axes[0].hist(df['text_length'], bins=50, color='skyblue', edgecolor='black')
axes[0].set_title('Distribution of Comment Lengths')
axes[0].set_xlabel('Number of Characters')
axes[0].set_ylabel('Frequency')

# Time-Series Density (Daily comment volume)
daily_volume = df.resample('D').size()
axes[1].plot(daily_volume, color='purple')
axes[1].set_title('Daily Reddit Comment Volume Over Time')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Number of Comments')
plt.tight_layout()
plt.show()"""),

        # Phase 2
        nbformat.v4.new_markdown_cell("## 6. Sophisticated NLP & Tokenization\nNext, we'll remove generic stopwords and build a frequency distribution of the most meaningful words representing the community's focus. We'll use NLTK to handle standard English stopwords."),
        nbformat.v4.new_code_cell("""import ast
from collections import Counter
from nltk.corpus import stopwords
import string

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)

# Custom function to safely evaluate tokenized lists and filter stopwords/punctuation
def filter_tokens(token_str):
    try:
        tokens = ast.literal_eval(token_str)
        # Keep alphabetic tokens not in stopwords, length > 2
        filtered = [word.lower() for word in tokens if word.lower() not in stop_words and word.isalpha() and len(word) > 2]
        return filtered
    except:
        return []

df['clean_tokens'] = df['tokenized'].apply(filter_tokens)

# Flatten list of lists to get comprehensive word frequencies
all_words = [word for tokens in df['clean_tokens'] for word in tokens]
word_freq = Counter(all_words)
common_words = word_freq.most_common(20)

words, counts = zip(*common_words)

plt.figure(figsize=(12, 6))
plt.bar(words, counts, color='coral')
plt.title('Top 20 Most Frequent Words in Community')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()"""),
        
        # Phase 3
        nbformat.v4.new_markdown_cell("## 7. Granular Sentiment Analysis (VADER Compound Scores)\nInstead of relying solely on exact bull/bear keyword matches, let's harness VADER's complete compound score, which contextualizes emotion, capitalization, and negation."),
        nbformat.v4.new_code_cell("""# Using the pre-calculated 'sia' (compound score)
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df['sia'], bins=40, color='teal', edgecolor='black')
ax.set_title('Distribution of VADER Compound Sentiment Scores')
ax.set_xlabel('Compound Sentiment Score (-1 to 1)')
ax.set_ylabel('Frequency')
plt.show()

# Resample daily SIA average and calculate a 7-day rolling Volatility
daily_sia = df['sia'].resample('D').mean()
sia_volatility = daily_sia.rolling('7D').std()

fig, ax1 = plt.subplots(figsize=(14, 5))
ax1.plot(daily_sia, color='darkgreen', label='Daily Avg Sentiment (SIA)', alpha=0.7)
ax1.plot(sia_volatility, color='red', label='7-Day Rolling Sentiment Volatility')
ax1.set_title('Daily Sentiment vs Sentiment Volatility (Uncertainty)')
ax1.set_ylabel('Score')
ax1.set_xlabel('Date')
ax1.legend()
plt.show()"""),

        # Phase 4
        nbformat.v4.new_markdown_cell("## 8. Financial Correlation Metrics\nTo provide sophisticated EDA for alternative data in finance, we must measure the statistical correlation between returns and sentiment. Does sentiment drive price, or does price drive sentiment?"),
        nbformat.v4.new_code_cell("""# Align BTC price and Daily Sentiment (inner join)
btc_aligned = btcusd.to_frame(name='Close')
btc_aligned.index = pd.to_datetime(btc_aligned.index).tz_localize(None)

# Let's align indices to TZ naive dates
daily_sia.index = pd.to_datetime(daily_sia.index).tz_localize(None)

# Merge datasets
combined_df = pd.merge(btc_aligned, daily_sia.to_frame(name='Daily_SIA'), left_index=True, right_index=True, how='inner')

# Calculate Logarithmic Returns for BTC to stabilize variance
combined_df['Log_Returns'] = np.log(combined_df['Close'] / combined_df['Close'].shift(1))

# Calculate daily change in sentiment (First Difference)
combined_df['SIA_Change'] = combined_df['Daily_SIA'].diff()

combined_df.dropna(inplace=True)

# Scatter plot: Sentiment Change vs BTC Returns
plt.figure(figsize=(8, 6))
plt.scatter(combined_df['SIA_Change'], combined_df['Log_Returns'], alpha=0.6, color='Navy')
plt.axhline(0, color='black', linestyle='--')
plt.axvline(0, color='black', linestyle='--')
plt.title('Daily Change in Sentiment vs. BTC Log Returns')
plt.xlabel('Change in Sentiment Score')
plt.ylabel('BTC Log Returns')
plt.grid(True, alpha=0.3)
plt.show()

# Statistical correlation coefficients
pearson_corr = combined_df[['SIA_Change', 'Log_Returns']].corr(method='pearson').iloc[0, 1]
spearman_corr = combined_df[['SIA_Change', 'Log_Returns']].corr(method='spearman').iloc[0, 1]

print(f"Pearson Correlation (Linear relationship): {pearson_corr:.4f}")
print(f"Spearman Correlation (Monotonic relationship): {spearman_corr:.4f}")
print("\\nInterpretation: Values near 0 indicate weak immediate daily correlation. Values moving towards 1 or -1 indicate significant predictive or reactive interactions.")""")
    ]
    
    nb.cells.extend(cells_to_add)
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print("Notebook successfully modified.")

if __name__ == '__main__':
    add_advanced_eda('/Users/anirudh/Documents/WQU/Course 02 600 FINMANCIAL DATA/GWP/Behavioral_Social_Media_EDA.ipynb')
