### Project Title
Enhancing Credit Score Prediction to Empower Financial Inclusion

**Author**
Vassilis Tsoubris

#### Executive summary

#### Rationale
Why should anyone care about this question?

#### Research Question
Can machine learning and AI models accurately predict individual credit scores based on financial data, thereby improving the decision-making process for lending institutions?

#### Data Sources
The primary dataset for this analysis will be sourced from Kaggle, specifically the "Credit Score Classification" dataset available at this [link](https://www.kaggle.com/datasets/parisrohan/credit-score-classification?select=train.csv).

This dataset contains historical financial data, including payment history, credit usage, and personal demographics, crucial for predicting credit scores. The target variable is the credit score of the customer ( 3 classes: Poor, Standard, Good ).
This dataset consists of 25 columns and contains 12500 customers. It is quite a challenging dataset as it needs advanced techniques to clean the data and transform its shape.

#### Methodology
The project will utilize a combination of supervised machine learning techniques, including but not limited to, Random Forests, Gradient Boosting Machines (GBM),  Logistic Regression, Decision Trees, Convolutional Neural Networks etc.. 
Preliminary data exploration and cleaning will be conducted using Python's pandas, SciPy and NumPy libraries. 
For model building and evaluation, Scikit-learn and Tensorflow will be the primary libraries, with a focus on cross-validation for model selection and metrics such as ROC-AUC, Accuracy, Precision, Recall, and F1 score to evaluate performance. 
For all the visualisations I will mainly use Seaborn and Matplotlib. 
Furthermore, I am researching more libraries to include like Yellowbrick for model interpretation and Optuna for more advanced hyperparameter optimisation.


#### Results
What did your research find?

#### Next steps
What suggestions do you have for next steps?
I would like to apply a model called ANFIS ( Artificial Neuro-Fuzzy Inference System ) which is a NN type.

#### Outline of project
Notebooks:
- [Notebook 1 - Initial Data Cleansing and Understanding](DataCleansing.ipynb)
- [Notebook 2 - EDA on Clean Data and final dataset](EDA_cleaned_data.ipynb)
- [Link to notebook 3]()

Python Scripts:
- [Custom functions used on Notebook 2](custom_functions/utility_functions.py) 


##### Contact and Further Information
Contact me through:
- [LinkedIn](https://www.linkedin.com/in/vtsoubris/)
