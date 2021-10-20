# imports
import pandas as pd
import numpy as np
import os
from env import user, host, password

from scipy import stats

from datetime import datetime
from math import sqrt

import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import register_matplotlib_converters

from matplotlib.ticker import StrMethodFormatter
from matplotlib.dates import DateFormatter

import statsmodels.api as sm
from statsmodels.tsa.api import Holt
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression, RFE


def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    It takes in a string name of a database as an argument.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
    
    
    
def new_cohort_logs_data():
    '''
    This function reads the 'cohorts' and 'logs' table data from the curriculum_logs database in 
    Codeup db into a df, writes it to a csv file, and returns the df.
    '''
    # Create SQL query.
    sql_query = """
                SELECT * 
                FROM logs
                LEFT JOIN cohorts ON cohorts.id = logs.cohort_id;
                """
                
                
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_connection('curriculum_logs'))
    
    return df



def get_cohort_logs_data():
    '''
    This function reads in titanic data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('cohort_logs.csv'):
        
        # If csv file exists, read in data from csv file.
        df = pd.read_csv('cohort_logs.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame.
        df = new_cohort_logs_data()
        
        # Write DataFrame to a csv file.
        df.to_csv('cohort_logs.csv')
        
    return df



def find_unique_values(df):
    for col in df.columns:
        print(f"Column: {col}")
        print("List of Values: ")
        print(df[col].unique())
        print("---------------------")
    return print('End')

def describe_data(df):
    print(df.isna().sum())
    print(df.info())
    print(df.describe())
    return print('End')

def dist_of_target(df, target, var):
    for c in df[var].unique():
        sns.displot(x=target, data=df[(df[var] == c)])
        plt.title(f'{c} {target} Distribution')
        plt.show()
    return print('End')


def add_dummies(df):
    dummy_df = pd.get_dummies(df[['Category',"Sub-Category","Segment"]])
    dummy_df = dummy_df.drop(columns= ['Category_Furniture'])
    df = pd.concat([df, dummy_df], axis=1)
    return df

def split(df):
    '''
    take in a DataFrame and return train, validate, and test DataFrames.
    return train, validate, test DataFrames.
    '''
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123)
    return train, validate, test

def make_vars(train, validate, test):
    
    target = "Profit"
    clist = ['Category_Office Supplies', 'Category_Technology',
       'Sub-Category_Accessories', 'Sub-Category_Appliances',
       'Sub-Category_Art', 'Sub-Category_Binders', 'Sub-Category_Bookcases',
       'Sub-Category_Chairs', 'Sub-Category_Copiers', 'Sub-Category_Envelopes',
       'Sub-Category_Fasteners', 'Sub-Category_Furnishings',
       'Sub-Category_Labels', 'Sub-Category_Machines', 'Sub-Category_Paper',
       'Sub-Category_Phones', 'Sub-Category_Storage', 'Sub-Category_Supplies',
       'Sub-Category_Tables', 'Segment_Consumer', 'Segment_Corporate',
       'Segment_Home Office']

    # split train into X (dataframe, only col in list) & y (series, keep target only)
    X_train = train[clist]
    y_train = train[target]
    y_train = pd.DataFrame(y_train)
    
    # split validate into X (dataframe, only col in list) & y (series, keep target only)
    X_validate = validate[clist]
    y_validate = validate[target]
    y_validate = pd.DataFrame(y_validate)

    # split test into X (dataframe, only col in list) & y (series, keep target only)
    X_test = test[clist]
    y_test = test[target]
    y_test = pd.DataFrame(y_test)
    
    return target, X_train, y_train, X_validate, y_validate, X_test, y_test

def make_metric_df(y, y_pred, model_name, metric_df):
    if metric_df.size ==0:
        metric_df = pd.DataFrame(data=[
            {
                'model': model_name, 
                'RMSE_validate': mean_squared_error(
                    y,
                    y_pred) ** .5,
                'r^2_validate': explained_variance_score(
                    y,
                    y_pred)
            }])
        return metric_df
    else:
        return metric_df.append(
            {
                'model': model_name, 
                'RMSE_validate': mean_squared_error(
                    y,
                    y_pred) ** .5,
                'r^2_validate': explained_variance_score(
                    y,
                    y_pred)
            }, ignore_index=True)

def linear_regression():

    lm = LinearRegression(normalize=True)

    lm.fit(X_train, y_train.Profit)

    y_train['profit_pred_lm'] = lm.predict(X_train)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.Profit, y_train.profit_pred_lm) ** (1/2)

    # predict validate
    y_validate['profit_pred_lm'] = lm.predict(X_validate)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.Profit, y_validate.profit_pred_lm) ** (1/2)

    return print("RMSE for OLS using LinearRegression\nTraining/In-Sample: ", rmse_train, 
                  "\nValidation/Out-of-Sample: ", rmse_validate)

def lassolars(a):
    # given a for alpha
    
    # create the model object
    lars = LassoLars(alpha=a)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series!
    lars.fit(X_train, y_train.Profit)

    # predict train
    y_train['profit_pred_lars'] = lars.predict(X_train)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.Profit, y_train.profit_pred_lars) ** (1/2)

    # predict validate
    y_validate['profit_pred_lars'] = lars.predict(X_validate)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.Profit, y_validate.profit_pred_lars) ** (1/2)

    return print("RMSE for Lasso + Lars\nTraining/In-Sample: ", rmse_train, 
      "\nValidation/Out-of-Sample: ", rmse_validate)

def linear_regression_test():

    lm = LinearRegression(normalize=True)

    lm.fit(X_train, y_train.Profit)

    y_train['profit_pred_lm'] = lm.predict(X_train)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.Profit, y_train.profit_pred_lm) ** (1/2)

    # predict validate
    y_validate['profit_pred_lm'] = lm.predict(X_validate)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.Profit, y_validate.profit_pred_lm) ** (1/2)
    
    # predict test
    y_test['profit_pred_lm'] = lm.predict(X_test)

    # evaluate: rmse
    rmse_test = mean_squared_error(y_test.Profit, y_test.profit_pred_lm) ** (1/2)

    return print("RMSE for OLS using LinearRegression\nTraining/In-Sample: ", rmse_train, 
                  "\nValidation/Out-of-Sample: ", rmse_validate,
                "\nTest/Out-of-Sample: ", rmse_test)

def viz_model(y_test, y_train):
    y_test_sample = y_test.sample(n=197)
    y_train_sample = y_train.sample(n=197)


    plt.figure(figsize=(16,8))
    plt.plot(y_test_sample.Profit, y_train_sample.profit_pred_mean, alpha=.5, color="gray", label='_nolegend_')
    plt.annotate("Baseline: Predict Using Mean", (16, 9.5))
    plt.plot(y_test_sample.Profit, y_test_sample.Profit, alpha=.5, color="blue", label='_nolegend_')
    plt.annotate("The Ideal Line: Predicted = Actual", (.5, 3.5), rotation=15.5)

    plt.scatter(y_test_sample.Profit, y_test_sample.profit_pred_lm, 
                alpha=.5, color="green", s=100, label="Model OLS")
    plt.legend()
    plt.xlabel("Actual Profit")
    plt.ylabel("Predicted Profit")
    plt.title("Best Model")
    plt.show()



