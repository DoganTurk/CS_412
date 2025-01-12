# Instagram Influencer Category Prediction

## Overview

This repository contains the code and resources for a project aimed at classifying Instagram influencers into predefined categories based on the content of their posts and profile information. The project leverages various machine learning and natural language processing (NLP) techniques.

The repository includes:

### Files
- **classification.py**: This Python script contains the core logic of the project, including:
  - Data loading and preprocessing functions for both training and test datasets.
  - Implementation of a custom Turkish text preprocessor, `TurkishTextPreprocessor`, that handles:
    - Irrelevant content removal (HTML, URLs, excessive whitespace).
    - Normalization of Turkish characters.
    - Lowercase conversion.
    - Tokenization using Zemberek.
    - Stop word removal.
    - Lemmatization.
    - Emoji to text conversion.
    - Hashtag and mention handling.
  - Data loading for the training dataset from `/content/drive/My Drive/Project/training-dataset.jsonl.gz` and the labels from `/content/drive/My Drive/Project/train-classification.csv`.
  - Implementation of TF-IDF vectorization and model training using several machine learning algorithms:
    - Support Vector Classifier (SVC) with hyperparameter tuning.
    - LightGBM classifier.
    - Logistic Regression with hyperparameter tuning.
    - Random Forest classifier with hyperparameter tuning.
    - Sentence Transformer embeddings with K-means clustering and a Random Forest classifier.
    - Multi-Layer Perceptron (MLP) classifier with SMOTE and without.
  - Evaluation metrics (accuracy, classification report, and confusion matrix).
  - Integration with Google Drive for data access and model persistence.

## Methodology

### Data Acquisition
The training dataset is loaded from a gzipped JSONL file, which contains user profile information and their posts. The label information is loaded from a separate CSV file. Data is split into training and test sets based on the availability of categories and labels in the training data, where if a username exists in the category labels it is considered train data, if not its considered as test data.

### Preprocessing
Preprocessing is critical for ensuring the quality and consistency of input data. The following steps were taken:
- **Text Normalization**:
  - Standardized Turkish-specific characters such as "ý" to "ı" and "þ" to "ş" to maintain linguistic accuracy.
- **Tokenization**:
  - Applied Zemberek's tokenizer to split text into meaningful tokens, handling punctuation and special symbols appropriately.
- **Stop-Word Removal**:
  - Utilized a curated list of Turkish stop-words, customized to exclude common but non-informative terms in the dataset.
- **Emoji and Mention Handling**:
  - Converted emojis to descriptive text to retain emotional and contextual nuances.
  - Managed mentions by either retaining, removing, or treating them as regular tokens, depending on the task requirements.
- **Lemmatization**:
  - Leveraged Zemberek's morphological analyzer to extract base forms of words, improving generalization and reducing sparsity.

### Feature Extraction
- **TF-IDF Vectorization**:
  - Used term frequency-inverse document frequency (TF-IDF) to represent text data as weighted numerical vectors.
  - Configured n-grams (up to bi-grams) to capture contextual relationships between words.
- **Embedding Generation**:
  - Applied SentenceTransformers to create dense, high-dimensional embeddings for textual data, capturing semantic meaning effectively.

### Classification
- **Model Selection and Training**:
  - Experimented with multiple classifiers (SVC, Random Forest, LightGBM, and Logistic Regression) to identify the best-performing model.
  - Balanced class weights to address the issue of class imbalance in the dataset.
- **Deep Learning Approach**:
  - Fine-tuned a multilingual RoBERTa model for sequence classification tasks, adapting it for the Turkish language.
- **Hyperparameter Optimization**:
  - Employed GridSearchCV to fine-tune model parameters, improving accuracy and generalization.

### Clustering
- **K-Means Clustering**:
  - Grouped similar text data into clusters using K-Means, with the number of clusters informed by the unique categories in the dataset.
- **Cluster Evaluation**:
  - Analyzed cluster purity by comparing majority categories within each cluster to true labels.

### Evaluation
- **Performance Metrics**:
  - Evaluated models using accuracy, precision, recall, F1-score, and confusion matrices.
- **Visualization**:
  - Plotted feature importance and confusion matrices to interpret model behavior and identify areas of improvement.

## Results

### Experimental Findings
- **Accuracy Metrics**:
  - **SVC**: Achieved validation accuracy of 67% with balanced class weights.
  - **LightGBM**: Recorded a validation accuracy of 63% with optimal hyperparameters and balanced class weights.
  - **RoBERTa**: 
    - We used FacebookAI/xlm-roberta-base and it achieved 70% accuracy.
    - Burakaytan/roberta-base-turkish-uncased reached 67% accuracy.
  - **RandomForest**: Reached a validation accuracy of 64% with balanced class weights.
  - **Logistic Regression**: Achieved validation accuracy of 68% with hyperparameter tuning and balanced class weights.
  - **CatBoost**: Recorded a validation accuracy of 62%.
  - **MLP**: Achieved validation accuracy of 65%, and with SMOTE it decreased to 62%.

**Results of SVC Classification**: SVC reached the best cross-validation accuracy among all models with a 67% accuracy rate and was used for ROUND3 classification.

**Results of RoBERTa Classification**: While not as robust as SVC, RandomForest, or Logistic Regression, RoBERTa achieved 70% accuracy on validation. Therefore, it was used for classification ROUND1 (FacebookAI/xlm-roberta-base) and ROUND2 (burakaytan/roberta-base-turkish-uncased).

## Instagram Like Count Prediction

### Overview

For the regression task of the project, multiple models were built in every round to achieve greater performance metrics. The features used were the same throughout the project. The preprocessing pipeline was designed to handle data cleaning, feature engineering, and user-level aggregation effectively. Key transformations were implemented, including logarithmic scaling of follower and comment counts, alongside categorical encoding of business account status.

### Round 1: 
In the first round, a dual-model system was developed to handle both regular and high-engagement accounts. The system demonstrated strong predictive performance, particularly for high-engagement outlier accounts.

### Round 2: 
In the second round, a clustering-based system was developed to handle different user engagement patterns. The system demonstrated enhanced predictive performance through the implementation of three distinct clusters.

### Round 3: 
In the third round, an enhanced clustering-based system was developed using a combination of DBSCAN and K-means algorithms to improve prediction accuracy. This iteration showed significant improvement in accuracy with refined clustering methods.

## Comparative Analysis

The evolution across rounds shows a clear progression in prediction accuracy:
- **Round 1** established the importance of separate handling for different engagement levels, with MSE values in the 0.3-0.8 range.
- **Round 2** refined this approach through clustering, though with higher MSE values, it provided more nuanced predictions across user segments.
- **Round 3** achieved the most precise predictions with MSE values in the 0.001-0.002 range, representing an order of magnitude improvement over previous rounds.

## Team Contributions

- **Hüseyin Doğan Türk (Classification)**:
  - Designed and implemented the text preprocessing pipeline.
  - Curated the Turkish stop-word list and emoji mappings.
  - Conducted deep learning experiments using RoBERTa.
  - Integrated SentenceTransformers and implemented clustering algorithms.
  - Evaluated cluster purity and refined embedding techniques.

- **Süleyman Berber (Classification)**:
  - Developed scripts for TF-IDF vectorization and training classification models.
  - Conducted hyperparameter tuning and model optimization.
  - Conducted deep learning experiments using RoBERTa.
  - Managed evaluation metrics and generated visualizations for results.

- **İlhan Sertelli (Regression)**:
  - Developed the dataset preprocessing.
  - Applied visualizations and clustering.
  - Applied cross-validation operations to improve performance.
  - Helped with feature selection.

- **Atacan Dilber (Regression)**:
  - Executed feature selection and correlation analysis.
  - Helped implement Random Forest and Ridge Regression models.
  - Generated and validated prediction outputs.
  - Assisted in applying K-means and DBSCAN clustering.
  - Developed preprocessing pipeline and implemented feature engineering.

