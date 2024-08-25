# Decoding the Language of Wine: Innovative Quality Prediction with NLP and kNN

## Project Overview

This project explores a novel approach to wine quality prediction by combining traditional machine learning techniques with natural language processing (NLP). By leveraging both numerical and textual data, we aim to enhance the accuracy of wine quality predictions, providing a comprehensive analysis of what makes a wine high or low quality.

### Objectives

- To predict wine quality (low/high) using the k-Nearest Neighbors (kNN) classification algorithm.
- To incorporate Natural Language Processing (NLP) to analyze textual descriptions of wines, enhancing the predictive model's understanding.
- To demonstrate the effectiveness of integrating multiple data modalities (numerical and text) for improved prediction outcomes.

## Project Structure

- data/: Contains the wine quality dataset and any additional textual data used for NLP.
- notebooks/: Jupyter Notebooks with detailed code and analysis steps.
- models/: Saved models and scripts for training and evaluation.
- results/: Output files, including visualizations and evaluation metrics.
- README.md: This file, providing an overview of the project.

## Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook
- Required Python libraries: pandas, numpy, matplotlib, scikit-learn, nltk (or any other NLP libraries you choose)

### Installation

1. Clone the repository:
    bash
    git clone https://github.com/yourusername/wine-quality-nlp-knn.git
    

2. Install the required libraries:
    bash
    pip install pandas numpy matplotlib scikit-learn nltk
    

3. Open the Jupyter Notebook:
    bash
    jupyter notebook
    

## Methodology

1. *Data Collection*: The primary dataset used is the Wine Quality dataset, which includes numerical features such as acidity, pH, and alcohol content. Additional textual data may be sourced from wine reviews or descriptions.

2. *Data Preprocessing*: 
   - Numerical features are cleaned and normalized.
   - Textual data is processed using NLP techniques, including tokenization, stopword removal, and vectorization.

3. *Feature Engineering*: Combining numerical features with NLP-extracted features to create a comprehensive feature set.

4. *Model Training*: 
   - The k-Nearest Neighbors (kNN) classifier is trained on the combined feature set.
   - Hyperparameter tuning is performed to optimize the model.

5. *Evaluation*: Model performance is evaluated using metrics such as accuracy, precision, recall, and F1-score. Various visualizations are created to illustrate and compare the model's performance.

## Results

- The integration of NLP features with traditional numerical features provides a more robust prediction model.
- Comparative analysis shows the improvement in prediction accuracy when incorporating text-based features.

## Conclusion

This project demonstrates that leveraging both numerical and textual data can significantly improve the prediction of wine quality. By decoding the 'language' of wine through NLP, we gain additional insights that traditional numerical analysis might miss.

## Contributors

- Ninitha Ram Mysore Shantha Ram
- Msc.Cyber Security
- 3121344@stud.srh-campus-berlin.de
