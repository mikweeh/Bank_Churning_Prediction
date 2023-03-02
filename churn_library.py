"""
Module for churn calculation.

Author: MCT
Date: March 2023
"""


# import libraries
import os
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


# Constants
OUTPUT_PATH = r"./output/"


def import_data(path):
    '''
    returns dataframe for the csv found at pth

    input:
        path: a path to the csv
    output:
        _: pandas dataframe
    '''
    return pd.read_csv(path)


def perform_eda(dtf):
    '''
    Perform eda on dataframe and save figures to images folder

    input:
        dtf: pandas dataframe

    output:
        None
    '''
    # dtf.head()
    # dtf.shape
    # dtf.isnull().sum()
    # dtf.describe()

    # Define a list of tuples for the plots and filenames
    plots = [
        ('Churn', 'Churn_histogram.png'),
        ('Customer_Age', 'Age_histogram.png'),
        ('Marital_Status', 'Marital_status_plot.png'),
        ('Total_Trans_Ct', 'Total_trans_Ct_plot.png'),
        ('Heatmap', 'Heatmap.png')]

    # Create output folder if it doesen't exist
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)

    # Generate the plots and save them
    for plot in plots:
        plt.figure(figsize=(20, 10))
        if plot[0] == 'Heatmap':
            sns.heatmap(dtf.corr(), annot=False, cmap='Dark2_r', linewidths=2)
        elif plot[0] == 'Marital_Status':
            dtf[plot[0]].value_counts('normalize').plot(kind='bar')
        elif plot[0] == 'Total_Trans_Ct':
            sns.histplot(dtf[plot[0]], stat='density', kde=True)
        else:
            dtf[plot[0]].hist()

        plt.savefig(OUTPUT_PATH + plot[1])
        plt.close('all')


def encoder_helper(dataframe, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
        dataframe: pandas dataframe
        category_lst: list of columns that contain categorical features
        response: string of response name

    output:
        dataframe: pandas dataframe with new columns
    '''

    # Define internal helper function to encode column
    def encode_column(dtf, col_name, col_new_name):
        col_groups = dtf.groupby(col_name)['Churn'].mean()
        dtf[col_new_name] = dtf[col_name].map(col_groups)

    # Check if "response" is a valid list
    if len(category_lst) != len(response):
        # logging "New columns get their default value"
        response = [cat + '_Churn' for cat in category_lst]

    # Apply encode_column() to each column in category_lst. dataframe is modified
    # inplace
    _ = [encode_column(dataframe, cat, resp)
         for cat, resp in zip(category_lst, response)]

    return dataframe


def perform_feature_engineering(dtf, response):
    '''
    Arrange input data for the training

    input:
        dtf: pandas dataframe
        response: string of response name [optional argument that could be
            used for naming variables or index y column]

    output:
        x_train: X training data
        x_test: X testing data
        y_train: y training data
        y_test: y testing data
    '''
    # Get list of numerical columns
    quant_columns = list(dtf.iloc[:, 4:].select_dtypes(
        include=['int64', 'float64']).columns)

    # Check if "response" is a valid list
    if len(quant_columns) != len(response):
        # Columns get their default name
        response = quant_columns

    # Define X and y with new names defined in "response"
    name_map = {old_name: response[i]
                for i, old_name in enumerate(quant_columns)}
    x_matrix = dtf.loc[:, [*quant_columns]].rename(columns=name_map)
    y_matrix = dtf['Churn']

    # Train test split
    x_train, x_test, y_train, y_test = train_test_split(
        x_matrix, y_matrix, test_size=0.3, random_state=42)

    return x_train, x_test, y_train, y_test


def classification_report_image(
        y_train,
        y_test,
        x_train,
        x_test,
        models):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
        y_train: training response values
        y_test:  test response values
        x_train: training input values
        x_test:  test input values
        models: dictionary containing the models ('rf' stands for random forest and 'lr'
            stands for logistic regression)

    output:
        None
    '''
    # Get predictions of random forest model
    y_train_preds_rf = models['rf'].predict(x_train)
    y_test_preds_rf = models['rf'].predict(x_test)

    # Get predictions of logistic regression model
    y_train_preds_lr = models['lr'].predict(x_train)
    y_test_preds_lr = models['lr'].predict(x_test)

    # Build report from Random Forest
    msg = 'Random Forest results\n'
    msg += 'Test results:\n'
    msg += classification_report(y_test, y_test_preds_rf)
    msg += '\n\nTrain results\n'
    msg += classification_report(y_train, y_train_preds_rf)

    # Building report from Logistic Regression
    msg += '\n\n\nLogistic Regression results\n'
    msg += 'Test results:\n'
    msg += classification_report(y_test, y_test_preds_lr)
    msg += '\n\ntrain results\n'
    msg += classification_report(y_train, y_train_preds_lr)

    # Save the report as an image
    _, ax1 = plt.subplots()
    ax1.text(0, 0, msg, fontsize=12, fontfamily='monospace')
    ax1.axis('off')
    plt.savefig(OUTPUT_PATH + 'Classification_report.png', bbox_inches='tight')


def feature_importance_plot(model, x_data):
    '''
    creates and stores the feature importances in pth
    input:
        model: model object containing feature_importances_
        x_data: pandas dataframe of X values

    output:
        fig: matplotlib figure with importances as bars
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    fig = plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)

    return fig


def train_models(x_train, y_train):
    '''
    Train and store models
    input:
        x_train: X training data
        y_train: y training data
    output:
        None
    '''
    # Grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']}

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)

    lrc.fit(x_train, y_train)

    # Create models folder if it doesen't exist
    out_path = r"./models/"
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    # Save best models
    joblib.dump(cv_rfc.best_estimator_, out_path + 'rfc_model.pkl')
    joblib.dump(lrc, out_path + 'logistic_model.pkl')


def save_results(x_train, x_test, y_train, y_test):
    '''
    Save results (plots) on disk
    input:
        x_train: X training data
        x_test: X testing data
        y_train: y training data
        y_test: y testing data
    output:
        None
    '''
    # Get the models
    models = dict()
    models['rf'] = joblib.load('./models/rfc_model.pkl')  # Random forest model
    models['lr'] = joblib.load(
        './models/logistic_model.pkl')  # Logistic regression

    # Create output folder if it doesen't exist
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)

    # Do the report
    classification_report_image(y_train, y_test, x_train, x_test, models)

    # Random Forest and Logistic Regression roc curves
    lrc_plot = plot_roc_curve(models['lr'], x_test, y_test)
    plt.figure(figsize=(15, 8))
    ax1 = plt.gca()
    plot_roc_curve(models['rf'], x_test, y_test, ax=ax1, alpha=0.8)
    lrc_plot.plot(ax=ax1, alpha=0.8)

    # Save the roc curve as a figure
    plt.savefig(OUTPUT_PATH + "ROC_curves.png")
    plt.close('all')

    # Shap plots
    explainer = shap.TreeExplainer(models['rf'])
    shap_values = explainer.shap_values(x_test)
    shap.summary_plot(shap_values, x_test, plot_type="bar")

    # Save the shap plot as a figure.
    plt.savefig(OUTPUT_PATH + "Tree_explainer.png")
    plt.close('all')

    # Feature importance plot
    img_plt = feature_importance_plot(models['rf'], pd.concat(
        [x_train, x_test], axis=0))

    # Save the importance plot as a figure
    img_plt.savefig(OUTPUT_PATH + "Importances.png")


# Full DEA pipeline example using the above functions
if __name__ == "__main__":

    # Load data
    INPUT_PTH = r"./data/bank_data.csv"
    data = pd.DataFrame(import_data(INPUT_PTH))

    # Create Churn column as a binary version of "Attrition_Flag", and insert
    # at position 0
    churn_column = data['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    data.insert(loc=0, column='Churn', value=churn_column)

    # Perform EDA
    perform_eda(data)

    # Get list of categorical columns
    cat_columns = list(data.iloc[:, 4:].select_dtypes(
        include=['object']).columns)

    # Encode the columns
    data = encoder_helper(data, cat_columns, [])

    # Feature engineering (extract numeric features and split datasets)
    xtrain, xtest, ytrain, ytest = perform_feature_engineering(data, [])

    # Training
    train_models(xtrain, ytrain)

    # Record results to files
    save_results(xtrain, xtest, ytrain, ytest)
