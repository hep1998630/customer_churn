"""
This library contains functions to find costumers who are likely to churn
Author: Mohamed Hafez
Date of developmen: Dec, 21, 2023

"""

# import libraries
import os
import joblib
import numpy as np
from sklearn.metrics import classification_report, plot_roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            imported_df: pandas dataframe
    '''
    imported_df = pd.read_csv(pth)
    imported_df['Churn'] = imported_df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return imported_df


def perform_eda(eda_df):
    '''
    perform eda on df and save figures to images folder
    input:
            eda_df: pandas dataframe

    output:
            None
    '''
    # Create Churn histogram
    plt.figure(figsize=(20, 10))
    eda_df['Churn'].hist()
    plt.savefig('images/eda/churn_histogram.png')

    # Create Customer Age histogram
    plt.figure(figsize=(20, 10))
    eda_df['Customer_Age'].hist()
    plt.savefig("images/eda/customer_age_histogram.png")

    # Marital Status barchart
    plt.figure(figsize=(20, 10))
    eda_df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig("images/eda/MS_bar.png")

    # Total Trans CT histogram
    plt.figure(figsize=(20, 10))
    sns.histplot(eda_df['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig("images/eda/TTCT_histogram.png")

    # Correlation plot
    plt.figure(figsize=(20, 10))
    sns.heatmap(eda_df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig("images/eda/correlation.png")


def encoder_helper(encoded_df, category_lst):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            imported_df: pandas dataframe
            category_lst: list of columns that contain categorical features

    output:
            df: pandas dataframe with new columns for
    '''
    for category in category_lst:

        # gender encoded column
        churn_encoded_list = []
        feature_groups = encoded_df.groupby(category).mean()['Churn']

        for val in encoded_df[category]:
            churn_encoded_list.append(feature_groups.loc[val])

        new_col_name = category + "_Churn"
        encoded_df[new_col_name] = churn_encoded_list

    return encoded_df


def perform_feature_engineering(df):
    '''
    input:
            df: pandas dataframe

    output:
            X_train: X training data
            X_test: X testing data
            y_train: y training data
            y_test: y testing data
    '''
    # Define output and initialize input
    y = df["Churn"]
    X = pd.DataFrame()

    # Define list of features to be encoded
    category_list = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']
    # Encode features using the helper function
    df_encoded = encoder_helper(df, category_list)

    # Define columns to be kept
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']
    # Define X before splitting data
    X[keep_cols] = df[keep_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
            None
    '''
    # Random forest report
    plt.figure(figsize=(5, 5))
    plt.rc('figure', figsize=(5, 5))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig("images/results/cls_report_RF.png", bbox_inches='tight')

    # Logistic regression report
    plt.figure(figsize=(5, 5))
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig("images/results/cls_report_LR.png", bbox_inches='tight')


def feature_importance_plot(
        model,
        X_data,
        output_pth="images/results/feature_imp_RF.png"):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
            None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    plt.savefig(output_pth)


def save_roc_curve(model_lr, model_RF, X_test, y_test):

    lrc_plot = plot_roc_curve(model_lr, X_test, y_test)
    # plots
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(
        model_RF.best_estimator_,
        X_test,
        y_test,
        ax=ax,
        alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig("images/results/ROC_curve.png", bbox_inches='tight')


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
            X_train: X training data
            X_test: X testing data
            y_train: y training data
            y_test: y testing data
    output:
            None
    '''

    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    classification_report_image(y_train, y_test,
                                y_train_preds_lr, y_train_preds_rf,
                                y_test_preds_lr, y_test_preds_rf)

    X = pd.concat([X_train, X_test], axis=0)

    # Plot feature importance for RF
    feature_importance_plot(
        cv_rfc.best_estimator_,
        X,
        output_pth="images/results/feature_imp_RF.png")
    save_roc_curve(lrc, cv_rfc, X_test, y_test)

    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')


if __name__ == "__main__":
    path = "data/bank_data.csv"
    # Read file into a dataframe
    df = import_data(path)

    # Perform exploratory data analysis and save the results in figures.
    perform_eda(df)

    # Perform feature engineering and split data.
    X_train, X_test, y_train, y_test = perform_feature_engineering(df)

    # Train models, and save training results.
    train_models(X_train, X_test, y_train, y_test)
