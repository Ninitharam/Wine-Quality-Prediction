# Decoding the Language of Wine: Innovative Quality Prediction with NLP and kNN

## Table of Contents

1. [Project Overview](#project-overview)
2. [Objectives](#objectives)
3. [Project Structure](#project-structure)
4. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
5. [Methodology](#methodology)
   - [Data Collection](#data-collection)
   - [Data Preprocessing](#data-preprocessing)
   - [NLP Integration](#nlp-integration)
   - [Feature Engineering](#feature-engineering)
   - [Model Training](#model-training)
   - [Evaluation](#evaluation)
6. [Results](#results)
7. [Conclusion](#conclusion)
9. [Contributors](#contributors)


## Project Overview

This project explores a novel approach to wine quality prediction by combining traditional machine learning techniques with natural language processing (NLP). By leveraging both numerical and textual data, we aim to enhance the accuracy of wine quality predictions, providing a comprehensive analysis of what makes a wine high or low quality.

### Objectives

- To predict wine quality (low/high) using the k-Nearest Neighbors (kNN) classification algorithm.
- To incorporate Natural Language Processing (NLP) to analyze textual descriptions of wines, enhancing the predictive model's understanding.
- To demonstrate the effectiveness of integrating multiple data modalities (numerical and text) for improved prediction outcomes.

## Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook
- Required Python libraries: pandas, numpy, matplotlib, scikit-learn, nltk, scipy (or any other NLP libraries you choose)

### Installation

1. Clone the repository:
   ```bash git clone https://github.com/yourusername/wine-quality-nlp-knn.git```

2. Install the required libraries:
  '''  bash
    pip install pandas numpy matplotlib scikit-learn nltk scipy
    '''

3. Open the Jupyter Notebook:
  '''  bash
    jupyter notebook
   ''' 

## Methodology

### Data Collection

The primary dataset used is the Wine Quality dataset, which includes numerical features such as acidity, pH, and alcohol content. Additional textual data is sourced from wine reviews or descriptions to provide a richer understanding of each wine's quality attributes.

### Data Preprocessing

- *Numerical Data*: Features are cleaned to handle missing values, normalized, and prepared for analysis.
- *Textual Data*: NLP techniques are applied to the textual descriptions to extract meaningful features.

### NLP Integration

NLP is used to process and analyze textual descriptions of wines. Here's how NLP is integrated into the workflow:

1. *Text Collection*: Gather text descriptions of each wine, which might include reviews, tasting notes, or product descriptions.

2. *Text Preprocessing*:
   - *Tokenization*: Splitting text into individual words or tokens.
   - *Stopword Removal*: Removing common words that do not add significant meaning (e.g., 'and', 'is', 'the').
   - *Stemming/Lemmatization*: Reducing words to their base or root form (e.g., "tasting" becomes "taste").

3. *Feature Extraction*:
   - *TF-IDF (Term Frequency-Inverse Document Frequency)*: Convert text data into numerical vectors that represent the importance of words in the text corpus.
   - *Word Embeddings*: Using techniques like Word2Vec or GloVe to capture semantic meanings and relationships between words.

4. *Integration with Numerical Data*: The textual features derived from NLP are combined with numerical features to form a comprehensive feature set. This combined dataset is then used for training the machine learning model.

### Feature Engineering

- Numerical features (e.g., acidity, pH, alcohol) and text-based features (from the textual descriptions) are combined into a unified feature set.
- This integrated feature set provides a holistic view of each wine's characteristics, leveraging both qualitative and quantitative aspects.

### Model Training

- The k-Nearest Neighbors (kNN) classifier is trained using the combined feature set.
- Hyperparameter tuning is conducted to find the best parameters for the kNN model, optimizing its performance for wine quality prediction.

### Evaluation

Model performance is evaluated using metrics such as accuracy, precision, recall, and F1-score. Visualizations and comparative analyses are used to illustrate the model's performance and the impact of integrating NLP features.

## Results

- The integration of NLP features with traditional numerical features leads to a more robust prediction model.
- Comparative analysis shows that incorporating text-based features improves prediction accuracy, highlighting the value of understanding textual nuances in wine descriptions.

## Conclusion

This project demonstrates that leveraging both numerical and textual data can significantly improve the prediction of wine quality. By decoding the 'language' of wine through NLP, we gain additional insights that traditional numerical analysis might miss.

## Contributors

- Ninitha Ram Mysore Shantha Ram
- Msc.Cybersecurity
- 3121344@stud.srh-campus-berlin.de
