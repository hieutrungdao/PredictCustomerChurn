# library doc string
'''
Modules for eda, feature engineering, training about customer churn.

Author: Hieu Trung Dao
Date: March 16 2023
'''

# import libraries
import os
from multiprocessing import Pool, cpu_count

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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


def import_data(pth):
    '''
    returns input_df for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    return pd.read_csv(pth)


def save_figure(subplot, fname, folder_path="images/eda"):
    '''
    save figure of matplotlib AxesSubplot

    input:
            subplot: matplotlib AxesSubplot or sklearn plot
            fname: str. File name of figure.
            folder_path: str. Path to folder store figure file.
    output:
            None
    '''
    save_path = os.path.join(folder_path, fname)
    fig = subplot.get_figure()
    fig.savefig(save_path)
    plt.clf()


def perform_eda(input_df):
    '''
    perform eda on df and save figures to images folder
    input:
            input_df: pandas dataframe

    output:
            None
    '''

    with Pool(processes=cpu_count()) as pool:
        pool.apply_async(
            save_figure, (input_df['Churn'].hist(), "churn_hist.png"))
        pool.apply_async(
            save_figure,
            (input_df['Customer_Age'].hist(),
             "customer_age_hist.png"))
        pool.apply_async(save_figure, (input_df.Marital_Status.value_counts(
            'normalize').plot(kind='bar'), "marital_status_bar.png"))
        pool.apply_async(
            save_figure,
            (sns.histplot(
                input_df['Total_Trans_Ct'],
                stat='density',
                kde=True),
                "total_trans_ct_distributions.png"))
        pool.apply_async(
            save_figure,
            (sns.heatmap(
                input_df.corr(),
                annot=False,
                cmap='Dark2_r',
                linewidths=2),
                "feature_heatmap.png"))

        pool.close()
        pool.join()


def get_churn(input_df):
    '''
    Get churn column in dataframe
    input:
            input_df: pandas dataframe
    output:
            input_df: pandas dataframe
    '''
    input_df['Churn'] = input_df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return input_df


def encode_category(input_df, cat, response="Churn"):
    '''
    encode each category column
    input:
            input_df: pandas dataframe
            cat: str. String of category
            response: string of response name
    output:
            input_df: pandas dataframe
    '''
    groups = input_df.groupby(cat).mean()[response]
    input_df[cat + "_" +
              response] = input_df[cat].apply(lambda val: groups.loc[val])
    return input_df[cat + "_" + response]


def encoder_helper(input_df, category_lst, response="Churn"):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            input_df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name

    output:
            input_df: pandas dataframe with new columns for
    '''
    items = [[input_df, cat, response] for cat in category_lst]

    with Pool(processes=cpu_count()) as pool:
        result = pool.starmap_async(encode_category, items)

        pool.close()
        pool.join()

    for result in result.get():
        input_df[result.name] = result

    return input_df


def perform_feature_engineering(input_df, response="Churn"):
    '''
    input:
              input_df: pandas dataframe
              response: string of response name
    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    input_df = encoder_helper(input_df, cat_columns)
    cat_churn_columns = [col + "_Churn" for col in cat_columns]
    x_dataframe = input_df[quant_columns + cat_churn_columns]
    y_dataframe = input_df[response]

    return train_test_split(
        x_dataframe,
        y_dataframe,
        test_size=0.3,
        random_state=42)


def classification_report_image(y_train,
                                y_test,
                                y_train_preds,
                                y_test_preds,
                                model):
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
    if isinstance(model, tuple):
        model_name = model[0].__class__.__name__
    else:
        model_name = model.__class__.__name__

    print(f"Results of {model}")

    test_report = classification_report(y_test, y_test_preds, output_dict=True)
    print('Test results')
    print(test_report)
    save_figure(sns.heatmap(pd.DataFrame(
        test_report).iloc[:-1, :].T, annot=True), f"{model_name}_test_report.png")

    train_report = classification_report(
        y_train, y_train_preds, output_dict=True)
    print('Train results')
    print(train_report)
    save_figure(sns.heatmap(pd.DataFrame(
        train_report).iloc[:-1, :].T, annot=True), f"{model_name}_train_report.png")


def feature_importance_plot(model, x_data, output_pth):
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
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]
    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])
    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth)
    plt.clf()


def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    models = {
        LogisticRegression(
            solver='lbfgs',
            max_iter=3000,
            n_jobs=cpu_count()),
        GridSearchCV(
            estimator=RandomForestClassifier(
                random_state=42,
                n_jobs=cpu_count()),
            param_grid=param_grid,
            cv=5)}

    for model in models:
        model.fit(x_train, y_train)

        if isinstance(model, GridSearchCV):
            y_train_preds = model.best_estimator_.predict(x_train)
            y_test_preds = model.best_estimator_.predict(x_test)
        else:
            y_train_preds = model.predict(x_train)
            y_test_preds = model.predict(x_test)

        classification_report_image(
            y_train,
            y_test,
            y_train_preds,
            y_test_preds,
            model
        )

    ax_subplot = plt.gca()
    for model in models:
        if isinstance(model, GridSearchCV):
            plot_roc_curve(
                model.best_estimator_,
                x_test,
                y_test,
                ax=ax_subplot,
                alpha=0.8)
        else:
            plot_roc_curve(
                model, x_test, y_test, ax=ax_subplot, alpha=0.8)
    save_figure(ax_subplot, "roc_curve.png", folder_path="images/eda")

    for model in models:
        if isinstance(model, GridSearchCV):
            feature_importance_plot(model, x_train, "feature_importances.png")

    for model in models:
        if isinstance(model, GridSearchCV):
            joblib.dump(model.best_estimator_, './models/rfc_model.pkl')
        else:
            joblib.dump(model, './models/logistic_model.pkl')


if __name__ == "__main__":

    df = import_data(r"./data/bank_data.csv")
    df = get_churn(df)
    perform_eda(df)
    train_feat, test_feat, train_label, test_label = perform_feature_engineering(
        df)
    train_models(train_feat, test_feat, train_label, test_label)
