import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import  make_column_transformer

def Preprocessor_function(data,target_variable = 'Credit_Score', cat_encoder = 'oh', excluded_features = None):
    """
    Preprocessor_function

    :param data: The input data to be preprocessed-dataframe.
    :param target_variable: The name of the target variable column. Default is 'Credit_Score'.
    :param cat_encoder: The type of categorical encoder to use. Options are 'ord' and 'oh'. Default is 'oh'.
    :param excluded_features: List of column names to be excluded from preprocessing. Default is None.
    :return: The preprocessor object.

    """
    if excluded_features:
        excluded_features = list(set(excluded_features) & set(data.columns))
        data = data.drop(excluded_features, axis=1)

    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.values.tolist()
    categorical_columns = \
        data.select_dtypes(include=['object']).drop(target_variable, axis = 1).columns.values.tolist()


    if cat_encoder == 'ord':
        # define the column transformer
        preprocessor = make_column_transformer(
            (StandardScaler(), numeric_columns),
            (OrdinalEncoder(), categorical_columns),
            remainder='drop'
        )

    elif cat_encoder == 'oh':
        preprocessor = make_column_transformer(
            (StandardScaler(), numeric_columns),
            (OneHotEncoder(drop = 'if_binary'), categorical_columns),
            remainder='drop'
        )
    else:
        return f"Please provide a cat_encoder: 'ord' or 'oh' "

    return preprocessor