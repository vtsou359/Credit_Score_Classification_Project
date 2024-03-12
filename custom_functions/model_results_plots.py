####### imports below: ######

# to hide all warnings:
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc
# yellowbrick's visualisations:
from yellowbrick.classifier import PrecisionRecallCurve, ClassPredictionError


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