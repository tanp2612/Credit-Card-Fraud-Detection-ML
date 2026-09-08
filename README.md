# My-Project
Credit Card Fraud Detection ML Project

# Problem Statement:  
The primary objective of this project is to design and implement machine learning models capable of accurately identifying fraudulent credit card transactions. The analysis leverages a customer-level dataset sourced from Kaggle, originally compiled through a collaborative research initiative between Worldline and the Machine Learning Group. This dataset contains 284,807 transactions, with a mere 492 instances of confirmed fraud. Given this extreme class imbalance, a critical component of the project involves applying advanced data handling and sampling techniques before training the predictive models.

# Business Context:  
Retaining profitable clients is a top priority for financial institutions, making the threat of banking fraud a critical operational concern. Beyond direct financial losses—which the Nilson Report projected to reach $30 billion globally by 2020—fraud heavily damages institutional credibility and customer trust.

As digital payment channels continue to expand, so do the sophisticated avenues for fraudulent activity. Consequently, deploying machine learning for fraud detection is no longer an optional upgrade but a baseline necessity. Automated, intelligent systems allow banks to proactively flag suspicious behavior, significantly reducing costly chargebacks, manual review times, and the false rejection of legitimate customer transactions.

# Defining Fraud Mechanisms:    

Credit card fraud encompasses any unauthorized use of a cardholder's information for illicit financial gain. While "skimming" (cloning data directly from the card's magnetic stripe) remains highly prevalent, bad actors employ various other methods, including:

    Altering or manipulating legitimate cards

    Manufacturing counterfeit cards

    Exploiting lost or stolen physical cards

    Executing deceptive telemarketing schemes 

# Dataset Description:  
  
The dataset captures two days of credit card transactions made by European cardholders in September 2013. It is severely imbalanced, with fraudulent cases representing just 0.172% of the total dataset.

To protect user confidentiality, the data underwent a Principal Component Analysis (PCA) transformation. As a result, 28 of the features (labeled V1 through V28) are obfuscated principal components. The only unaltered features are:

    Time: The seconds elapsed between the first recorded transaction and subsequent transactions.

    Amount: The monetary value of the transaction.

    Class: The target variable, where 1 denotes a fraudulent transaction and 0 indicates a legitimate one.  

# Project Pipeline:  

The technical workflow is structured into five distinct phases:

    Data Familiarization: Loading the dataset and examining feature distributions to identify the variables most crucial for predictive modeling.

    Exploratory Data Analysis (EDA): Conducting univariate and bivariate analyses to uncover underlying patterns. Since most features are already PCA-transformed, standard Z-scaling may not be necessary. However, identifying and addressing data skewness remains vital to ensure optimal model training.

    Data Splitting & Validation: Segregating the data into training and testing sets to evaluate model generalization on unseen data. Given the extreme minority class, stratified techniques like k-fold cross-validation are essential to guarantee that fraudulent cases are adequately represented across all validation folds.

    Model Development & Tuning: Experimenting with various machine learning algorithms and optimizing their hyperparameters. A key focus here is utilizing robust sampling techniques (such as oversampling the minority class) to improve the algorithm's ability to learn from the imbalanced data.

    Performance Evaluation: Assessing the final models using metrics tailored to imbalanced datasets. Because the business cost of missing a fraudulent transaction (False Negative) is incredibly high, the chosen evaluation metrics prioritize the accurate identification of the positive class rather than relying on overall baseline accuracy.