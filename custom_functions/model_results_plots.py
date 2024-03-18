####### imports below: ######

# to hide all warnings:
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
# yellowbrick's visualisations:
from yellowbrick.classifier import PrecisionRecallCurve, ClassPredictionError
import pandas as pd
import seaborn as sns


######## functions below: #########


def multiclass_roc_auc_score_and_plot(model, X_test, y_test, model_name = 'Logistic Regression', plot_font_size = 13):
    """
    Compute the ROC AUC score for each class and plot the ROC curves.
    ( this function can be applied only for models that use predict_proba() )

    :param model: The trained classification model.
    :param X_test: The test set features.
    :param y_test: The test set targets.
    :param model_name: The name of the model (default is 'Logistic Regression').
    :param plot_font_size: The font size for the plot (default is 13).
    :return: Dictionary of ROC AUC scores for each class.
    """
    y_prob = model.predict_proba(X_test)

    # Binarize the labels
    lb = LabelBinarizer()
    y_test_bin = lb.fit_transform(y_test)

    # Get class names from your fitted model
    class_names = model.classes_

    n_classes = y_test_bin.shape[1]

    #to Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Plot of a ROC curve for a specific class
    plt.figure()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        plt.plot(fpr[i], tpr[i],
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(class_names[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=plot_font_size-1)
    plt.ylabel('True Positive Rate', fontsize=plot_font_size-1)
    plt.title('Receiver Operating Characteristic (ROC) of\n'+model_name+' Model (multi-class)',
              fontsize= plot_font_size)
    plt.legend(loc="lower right")
    plt.show()

    return roc_auc







def class_prediction_error_plot(X_train, y_train, X_test, y_test, estimator, is_fitted = True):
    """
    Plot the class prediction error for a given classifier.

    :param X_train: The training features.
    :param y_train: The training labels.
    :param X_test: The test features.
    :param y_test: The test labels.
    :param estimator: The classifier model to be visualized.
    :param is_fitted: (optional) Whether the estimator is already fitted. Default is True.
    :return: The visualizer object.

    """
    # Instantiate the classification model and visualizer
    visualizer = ClassPredictionError(estimator,
                                      classes = estimator.classes_,
                                      is_fitted = is_fitted) #do not fit the estimator again

    # Fit the training data to the visualizer (does not actually fit the estimator)
    visualizer.fit(X_train, y_train)

    # Evaluate the model on the test data
    visualizer.score(X_test, y_test)

    # Draw visualization
    visualizer.show()

    return visualizer








def precision_recall_curve_plot(X_train, y_train, X_test, y_test,
                                estimator,
                                line_per_class = True,
                                is_fitted = False,
                                plot_f1_curves=True,):
    """

    This function `precision_recall_curve_plot` plots the precision-recall curve for a given binary or multiclass classification model.

    :param X_train: The feature matrix of the training set.
    :param y_train: The target labels of the training set.
    :param X_test: The feature matrix of the test set.
    :param y_test: The target labels of the test set.
    :param estimator: The classification estimator model to evaluate.
    :param line_per_class: Whether to plot separate curves for each class (default=True). If set to False, a single curve will be plotted for all classes.
    :param is_fitted: Set to True if the estimator model is already fitted (default=False).
    :param plot_f1_curves: Whether to plot iso-F1 curves (default=True).
    :return: The PrecisionRecallCurve visualizer object.
    """
    if line_per_class:
        per_class= True
        micro= False
    else:
        per_class= False
        micro= True

    # Create the visualizer, fit, score, and show it
    viz = PrecisionRecallCurve(
        estimator,
        classes = estimator.classes_,
        per_class=per_class,
        micro=micro,
        is_fitted=is_fitted,
        force_model= True,
        iso_f1_curves= plot_f1_curves,
        cmap="Set1"
    )
    viz.fit(X_train, y_train)
    viz.score(X_test, y_test)
    viz.show()

    return viz







def all_models_metrics(grid_models, model_names, X_train, X_test, y_train, y_test):
    """
    :param grid_models: list of grid search models
    :param model_names: list of model names corresponding to grid models
    :param X_train: training features
    :param X_test: testing features
    :param y_train: training labels
    :param y_test: testing labels
    :return: DataFrame containing metrics for each model
    """
    # Initialize an empty dictionary to store metrics
    metrics_grid = {}

    # Should be same length for both lists
    assert len(grid_models) == len(model_names), "Mismatch in length of 'grid_models' and 'model_names'"

    for grid_model, model_name in zip(grid_models, model_names):
        results = pd.DataFrame(grid_model.cv_results_)

        # Extract best estimator and training time
        best_estimator = grid_model.best_estimator_
        training_time = results[results['params'] == grid_model.best_params_]['mean_fit_time'].values[0]

        # Make predictions
        y_train_pred = best_estimator.predict(X_train)
        y_test_pred = best_estimator.predict(X_test)

        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        recall = recall_score(y_test, y_test_pred, zero_division=0, average = 'micro')
        precision = precision_score(y_test, y_test_pred, zero_division=0, average = 'micro')
        f1 = f1_score(y_test, y_test_pred, zero_division=0, average = 'micro')

        # Store metrics in the dictionary
        metrics_grid[model_name] = {
            'training_time': training_time,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'recall micro-average': recall,
            'precision micro-average': precision,
            'f1_score micro-average': f1
        }

    # Convert dictionary to DataFrame and return
    df_metrics_grid = pd.DataFrame.from_dict(metrics_grid, orient='index')
    return df_metrics_grid




def plot_confusion_matrix(grid_search, X_test, y_test):
    # Use the best estimator from grid_lr to predict test set results
    y_pred = grid_search.best_estimator_.predict(X_test)

    # Calculate precision and recall
    precision = precision_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')


    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)


    # Plot confusion matrix
    plt.figure(figsize=(6, 6))  # Resize plot to allow space for formula and scores
    sns.heatmap(cm, annot=True, fmt='', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix of the Best Estimator: '+grid_search.best_estimator_.named_steps['estimator'].__class__.__name__)


    # Add the formulas for Precision and Recall at the bottom of the plot
    plt.text(1.55, 0.9,
             r'$Precision = \frac{TP}{TP + FP}$' + '\n' + f'Score: {precision:.2f}',
             horizontalalignment='center',
             verticalalignment='center',
             transform = plt.gca().transAxes)

    plt.text(1.55, 0.7,
             r'$Recall = \frac{TP}{TP + FN}$' + '\n' + f'Score: {recall:.2f}',
             horizontalalignment='center',
             verticalalignment='center',
             transform = plt.gca().transAxes)

    # Adjust the layout to make room for the formulas
    plt.subplots_adjust(bottom=0.2)
    plt.show()