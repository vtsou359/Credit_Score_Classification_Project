import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

from sklearn.preprocessing import LabelEncoder

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



def correlation_with_target(data, feature, corr_method = 'kendall'):
    """
    Calculate the correlation (method = kendall) between a feature and the target variable.

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
    correlation = df[[feature, 'Credit_Score']].corr(method = corr_method).iloc[0, 1]

    return correlation



def find_correlated_features(df, threshold, corre_method = 'kendall'):
    """
    Computes correlation of each feature with target 'Credit_Score' and keeps features with correlation >= threshold.

    Params:
    df : pandas.DataFrame, The input dataframe
    threshold : Correlation threshold, only correlations >= threshold are kept

    Returns:
    pandas.core.series.Series, Features that have correlation >= threshold with 'Credit_Score'.
    """
    correlated_features = {}

    # Iterating over each column(feature) in the DataFrame
    for feature in df.columns:
        # Exclude the target variable 'Credit_Score'
        if feature != 'Credit_Score':
            correlation = correlation_with_target(df, feature, corr_method = corre_method)
            # Check if correlation value is numeric (i.e., the function didn't return an error message)
            if isinstance(correlation, (float, int)):
                # Check if correlation is greater than or equal to threshold
                if abs(correlation) >= threshold:
                    correlated_features[feature] = correlation

    # Converting the dictionary to a pandas Series for better visual output
    return pd.Series(correlated_features).sort_values(ascending=False)





def create_plots_box_violin(data):
    for column in data.columns:
        if column != 'Credit_Score':  # we don't want to make a plot for the target variable
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))

            # Boxplot
            sns.boxplot(x='Credit_Score', y=column, data=data, ax=axes[0], palette=['#F05941','#BE3144','#872341'], hue = 'Credit_Score')
            axes[0].set_title(f'Boxplot of {column}')  # Set title

            # Violin plot
            sns.violinplot(x='Credit_Score', y=column, data=data, ax=axes[1], palette=['#F05941','#BE3144','#872341'], hue = 'Credit_Score')
            axes[1].set_title(f'Violin plot of {column}')  # Set title

            plt.tight_layout()  # For better spacing between subplots
            plt.show()  # Show the plots



def plot_categorical_stacked(df, target = 'Credit_Score',excluded_feats=['Type_of_Loan'], percentage=False, color_map = ['#F05941','#BE3144','#872341']):
    """
    :param df: The Pandas DataFrame containing the categorical data.
    :param target: The target variable column name.
    :param excluded_feats: A list of column names to be excluded from the analysis.
    :param percentage: Boolean flag indicating whether to display counts or proportions in the plot.
    :return: None

    This method plots a stacked bar plot to visualize the distribution of categorical variables in relation to the target variable. It takes the following parameters:

    - df: The Pandas DataFrame containing the categorical data.
    - target: The target variable column name. By default, it is set to 'Credit_Score'.
    - excluded_feats: A list of column names to be excluded from the analysis. By default, it excludes the 'Type_of_Loan' column.
    - percentage: A boolean flag indicating whether to display the counts or proportions in the plot. By default, it is set to False.

    The method works by grouping the data by the target and categorical columns, calculating counts or proportions if specified, and plotting the stacked bar plot using matplotlib. The x
    *-axis labels are rotated for readability, and the plot is displayed using plt.show(). The title and y-label are set based on the values of the percentage flag.

    Note: This method requires the matplotlib and pandas libraries to be installed.
    """

    # creating a colormap to be used:
    colors = color_map
    my_cmap = ListedColormap(colors, name="my_cmap")


    categorical_columns = df.drop(excluded_feats, axis=1).select_dtypes(['object', 'category']).columns.tolist()

    if target in categorical_columns:
        categorical_columns.remove(target)

    for column in categorical_columns:
        counts = df.groupby([target, column]).size().unstack(target)

        if percentage:
            # Calculate the proportions
            counts = counts.apply(lambda x: (x / x.sum())*100, axis=1)

        # Plot
        counts.plot(kind='bar', stacked=True, figsize=(10, 6), colormap= my_cmap)
        if percentage:
            plt.title(f'Stacked bar plot of proportions - {column} per {target} class')
            plt.ylabel('Proportion')
        else:
            plt.title(f'Stacked bar plot - {column} per {target} class')
            plt.ylabel('Count')
        plt.xticks(rotation=45)  # Rotate x-axis labels for readability
        plt.tight_layout()
        plt.show()
