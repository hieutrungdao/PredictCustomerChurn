'''
Test modules for eda, feature engineering, training about customer churn.

Author: Hieu Trung Dao
Date: March 16 2023
'''

import os
import logging
import churn_library as cls

import matplotlib
matplotlib.use('Agg')

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


os.environ['QT_QPA_PLATFORM'] = 'offscreen'


cat_columns = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]

quant_columns = [
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
    'Avg_Utilization_Ratio'
]


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        dataframe = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err
    return dataframe


def test_save_figure(save_figure, dataframe):
    '''
    test perform eda function
    '''
    try:
        save_figure(dataframe['Churn'].hist(), "churn_hist.png")
        logging.info("Testing save_figure: SUCCESS")
    except Exception as err:
        logging.error("Testing save_figure: ERROR")
        raise err

    try:
        assert os.path.exists("images/eda/churn_hist.png")
    except AssertionError as err:
        logging.error(
            "Testing save_figure: The figure file doesn't exist")
        raise err


def test_eda(perform_eda, input_df):
    '''
    test perform eda function
    '''
    try:
        perform_eda(input_df)
        logging.info("Testing perform_eda: SUCCESS")
    except Exception as err:
        logging.error("Testing perform_eda: ERROR")
        raise err

    try:
        assert os.path.exists("images/eda/churn_hist.png")
        assert os.path.exists("images/eda/customer_age_hist.png")
        assert os.path.exists("images/eda/marital_status_bar.png")
        assert os.path.exists("images/eda/total_trans_ct_distributions.png")
        assert os.path.exists("images/eda/feature_heatmap.png")
    except AssertionError as err:
        logging.error(
            "Testing save_figure: The figure file doesn't exist")
        raise err


def test_get_churn(get_churn, input_df):
    '''
    test get churn function
    '''
    try:
        input_df = get_churn(input_df)
        logging.info("Testing get_churn: SUCCESS")
    except Exception as err:
        logging.error("Testing get_churn: ERROR")
        raise err

    try:
        assert 'Churn' in input_df.columns
    except AssertionError as err:
        logging.error(
            "Testing get_churn: Churn column doesn't exist")
        raise err

    try:
        for _, row in input_df.iterrows():
            if row["Attrition_Flag"] == "Existing Customer":
                assert row["Churn"] == 0
            else:
                assert row["Churn"] == 1
    except AssertionError as err:
        logging.error(
            "Testing get_churn: Wrong value of Churn")
        raise err

    return input_df


def test_encode_category(encode_category, input_df):
    '''
    test get churn function
    '''
    try:
        input_df = encode_category(input_df, "Gender")
        logging.info("Testing encode_category: SUCCESS")
    except Exception as err:
        logging.error("Testing encode_category: ERROR")
        raise err

    try:
        assert 'Gender_Churn' == input_df.name
    except AssertionError as err:
        logging.error(
            "Testing encode_category: Gender_Churn column doesn't exist")
        raise err

    return input_df


def test_encoder_helper(encoder_helper, input_df, category_lst):
    '''
    test encoder helper
    '''
    try:
        input_df = encoder_helper(input_df, category_lst)
        logging.info("Testing encoder_helper: SUCCESS")
    except Exception as err:
        logging.error("Testing encoder_helper: ERROR")
        raise err

    try:
        for cat in category_lst:
            assert cat+"_Churn" in input_df.columns
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: Encoded columns don't exist")
        raise err


def test_perform_feature_engineering(perform_feature_engineering, input_df):
    '''
    test perform_feature_engineering
    '''
    try:
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            input_df)
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except Exception as err:
        logging.error("Testing perform_feature_engineering: ERROR")
        raise err

    try:
        assert "Churn" not in x_train.columns
        assert "Churn" not in x_test.columns
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: Wrong split")
        raise err

    return x_train, x_test, y_train, y_test


def test_train_models(train_models, x_train, x_test, y_train, y_test):
    '''
    test train_models
    '''
    try:
        train_models(x_train, x_test, y_train, y_test)
        logging.info("Testing train_models: SUCCESS")
    except Exception as err:
        logging.error("Testing train_models: ERROR")
        raise err

    try:
        assert os.path.exists("./models/rfc_model.pkl")
        assert os.path.exists("./models/logistic_model.pkl")
    except AssertionError as err:
        logging.error(
            "Testing train_models: Not found model file")
        raise err

    try:
        assert os.path.exists("images/results/roc_curve.png")
    except AssertionError as err:
        logging.error(
            "Testing train_models: Not found score figure file")
        raise err


if __name__ == "__main__":
    df = test_import(cls.import_data)
    df = test_get_churn(cls.get_churn, df)

    test_save_figure(cls.save_figure, df)
    test_eda(cls.perform_eda, df)
    test_encode_category(cls.encode_category, df)
    test_encoder_helper(cls.encoder_helper, df, cat_columns)

    train_feat, test_feat, train_label, test_label = test_perform_feature_engineering(
        cls.perform_feature_engineering,
        df
    )
    test_train_models(
        cls.train_models,
        train_feat, test_feat, train_label, test_label
    )
