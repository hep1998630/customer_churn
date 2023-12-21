"""
Testing script for functions defined in churn_library.py
Author: Mohamed Hafez
Date of developmen: Dec, 21, 2023 
"""
import os
import logging
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(pth):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        imported_df = cls.import_data(pth)
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert imported_df.shape[0] > 0
        assert imported_df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err
    return imported_df


def test_eda(eda_df):
    '''
    test perform eda function
    '''
    cls.perform_eda(eda_df)
    number_of_files = len(os.listdir('./images/eda'))
    try:
        assert number_of_files == 5
        logging.info("Testing perform_eda: SUCCESS")
    except AssertionError as err:
        logging.error(
            f"Testing perform_eda: Expected 5 files found {number_of_files}")
        raise err


def test_encoder_helper(encoder_df):
    '''
    test encoder helper
    '''
    feature_name = "Gender"
    encoded_df = cls.encoder_helper(encoder_df, [feature_name])
    encoded_feat_name = feature_name + "_Churn"
    try:

        assert encoded_feat_name in encoded_df.columns
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: could not find the encoded feature")
        raise err


def test_perform_feature_engineering(feats_df):
    '''
    test perform_feature_engineering
    '''
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
    X_train_feats, X_test_feats, y_train_feats, y_test_feats = cls.perform_feature_engineering(feats_df)

    try:
        assert all(keep_cols) == all(X_train_feats.columns)
        assert all(keep_cols) == all(X_test_feats.columns)
        assert len(X_test_feats) == len(y_test_feats)
        assert len(X_train_feats) == len(y_train_feats)
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: Failed, some columns are not as expected.")
        raise err


def test_train_models(train_models, df):
    '''
    test train_models
    we will check if proper figures and models are saved opon training.
    '''
    X_train_feats, X_test_feats, y_train_feats, y_test_feats = cls.perform_feature_engineering(df)
    train_models(X_train_feats, X_test_feats, y_train_feats, y_test_feats)
    try:
        assert os.path.exists("images/results/feature_imp_RF.png")
        assert os.path.exists("images/results/ROC_curve.png")
        assert os.path.exists("images/results/cls_report_RF.png")
        assert os.path.exists("images/results/cls_report_LR.png")
        assert os.path.exists("models/logistic_model.pkl")
        assert os.path.exists("models/rfc_model.pkl")
        logging.info(
            "Testing test_train_models: SUCCESS >> all figures are created successfully")

    except AssertionError as err:
        logging.error(
            "Testing test_train_models: FAILED, some figures are missing")
        raise err


if __name__ == "__main__":
    data_pth = "./data/bank_data.csv"
    df = test_import(data_pth)
    # print(df["Churn"])
    test_eda(df)
    test_encoder_helper(df)
    test_perform_feature_engineering(df)
    test_train_models(cls.train_models, df)
