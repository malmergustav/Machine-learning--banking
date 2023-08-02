# Bank Marketing Analysis

This project aims to analyze bank marketing data and predict whether a customer will subscribe to a term deposit (y = 'yes') or not (y = 'no'). The code covers data cleaning, exploratory data analysis (EDA), model selection, hyperparameter tuning, and model evaluation.

## Data

The project uses the "bank-additional-full.csv" dataset, which contains various features related to bank marketing campaigns. The dataset is loaded into a Pandas DataFrame named `df`.

## Data Cleaning

The data cleaning process includes dropping the "duration" column since it affects the outcome too much and does not create any business value. Categorical columns like "job," "marital," "education," "default," "housing," and "loan" are cleaned by removing rows with "unknown" or "illiterate" values.

## Exploratory Data Analysis (EDA)

EDA is performed to gain insights into the data. Various visualizations are created, including histograms for the age, campaign, previous contact, employee variation rate, consumer price index, and Euribor interest rate. Additionally, a pair plot is generated to explore the relationships between numeric features.

## Model Selection

Three classifiers are tested for the prediction task: RandomForestClassifier, SVC (Support Vector Classifier), and LogisticRegression. The classifiers are trained and evaluated using the F1-score metric.

## Hyperparameter Tuning

Hyperparameter tuning is performed on the RandomForestClassifier, SVC, and LogisticRegression models using GridSearchCV to find the best parameters.

## Model Evaluation

The RandomForestClassifier with the best hyperparameters is chosen as the final model. It is refitted on all data and evaluated on the test set. Metrics such as accuracy, precision, recall, and F1-score are computed and visualized using bar plots.

## Files

The project contains the following files:

- `bank-additional-full.csv`: The dataset used for analysis.
- `age_distribution.png`: Histogram of the age distribution of contacted customers.
- `campaign_distribution.png`: Histogram of the distribution of the number of contacts with customers during this campaign.
- `previous_contacts.png`: Histogram of the distribution of the number of previous contacts with customers.
- `emp_distr.png`: Histogram of the employee variation rate.
- `cons_price.png`: Histogram of the consumer price index.
- `euribor.png`: Histogram of the Euribor interest rate.
- `model_performance.png`: Bar plot of the model's performance metrics (accuracy, precision, recall, and F1-score).
- `confusion_matrix.png`: Visualization of the confusion matrix.

## Getting Started

To run the code and reproduce the analysis, follow these steps:

1. Clone the repository to your local machine.
2. Install the required libraries: pandas, seaborn, matplotlib, scikit-learn.
3. Make sure the dataset "bank-additional-full.csv" is in the same directory as the notebook.
4. Run the notebook "Bank_marketing.ipynb" to execute the code and generate visualizations.
5. Review the results in the notebook and explore further if needed.

Feel free to modify the code or add more analyses based on your preferences or business requirements.

If you have any questions or need further assistance, please let me know!
