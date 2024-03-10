import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

def plot_triangular_heatmap(dataframe):
    """
    Plots a triangular correlation heatmap using the seaborn library.
    :param dataframe: A pandas DataFrame containing the data for which the correlation heatmap needs to be plotted.
    :return: None
    """
    plt.figure(figsize=(16, 6))
    #plt.grid(visible = False)
    mask = np.triu(np.ones_like(dataframe.corr()))
    heatmap = sns.heatmap(dataframe.corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')
    heatmap.set_title('Triangle Correlation Heatmap', fontdict={'fontsize':18}, pad=16);



def correlation_with_target(data, feature):
    """
    Calculate the correlation between a feature and the target variable.

    :param data: The input DataFrame containing the data.
    :type data: pandas.DataFrame
    :param feature: The name of the feature to calculate correlation with the target variable.
    :type feature: str
    :return: The correlation between the feature and the target variable.
    :rtype: float or str
    """
    # Check if feature exists in DataFrame
    if feature not in data.columns:
        return f"Feature {feature} not found in DataFrame"

    # Prepare a copy of DataFrame to avoid modifying original data
    df = data.copy()

    # Check target column exists
    if 'Credit_Score' not in df.columns:
        return "Target column 'Credit Score' not found in DataFrame"

    # Check if target variable is categorical, if yes then convert it using Label Encoder
    if df['Credit_Score'].dtype == 'object':
        df['Credit_Score'] = LabelEncoder().fit_transform(df['Credit_Score'])

    # Compute correlation
    correlation = df[[feature, 'Credit_Score']].corr().iloc[0, 1]

    return correlation




#%%
