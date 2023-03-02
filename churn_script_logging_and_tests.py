'''
Testing script for "churn_library.py" file

This testing script tests the full pipeline. Before running the script
be sure that the file "bank_data.csv" is in the folder "data" in the
project directory

Author: Mike
Date: March, 2023
'''

# Imports
import os
import logging
import pandas as pd
import churn_library as cls


# Create log folder if it doesen't exist
LOG_FOLDER = r'./logs/'
if not os.path.exists(LOG_FOLDER):
    os.mkdir(LOG_FOLDER)


# Logging setup
logging.basicConfig(
    filename=LOG_FOLDER + 'churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


# Constants
INPUT_PATH = r"./data/bank_data.csv"
OUTPUT_PATH = r"./output/"


# Test functions
def test_import_data(dtf):
    '''
    Test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df1 = pd.DataFrame(cls.import_data(INPUT_PATH))
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

    try:
        assert df1.shape == dtf.shape
        logging.info("Shape of the loaded dataframe: %s. SUCCESS", df1.shape)
    except AssertionError as err:
        logging.error(
            "Testing import_data: Error in shape of the dataframe. %s",
            df1.shape)
        raise Exception("Error in shape of the dataframe") from err

    return df1


def test_perform_eda(dtf):
    '''
    Test for function perform_eda()
    '''
    # Previous arrangement to do the EDA
    # Create Churn column as a binary version of "Attrition_Flag", and insert
    # at position 0
    churn_column = dtf['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    dtf.insert(loc=0, column='Churn', value=churn_column)

    # Test if EDA is performed
    try:
        cls.perform_eda(dtf)
        logging.info("EDA performed. SUCCESS")
    except BaseException:
        logging.error("Testing perform_eda(). Error performing EDA")
        raise

    # Test that output files are created
    expected_files = [
        'Churn_histogram.png',
        'Age_histogram.png',
        'Marital_status_plot.png',
        'Total_trans_Ct_plot.png',
        'Heatmap.png']
    try:
        for filename in expected_files:
            assert os.path.isfile(OUTPUT_PATH + filename)
        logging.info("All expected image files were properly created. SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing perform_eda(). Not all the expected files have been created. %s",
            err)
        raise err

    return dtf


def test_encoder_helper(dtf):
    '''
    Test for funtion encoder_helper()
    '''
    # Get list of categorical columns
    cat_cols = list(dtf.iloc[:, 4:].select_dtypes(include=['object']).columns)
    num_cols = dtf.shape[1]
    num_cat = len(cat_cols)

    # Encode the columns
    try:
        dtf = cls.encoder_helper(dtf, cat_cols, [])
        logging.info("Categorical columns encoded. SUCCESS")
    except Exception as err:
        logging.error(
            "Testing encoder_helper(). Some error ocurred. %s", err)
        raise err

    # Check that all new columns have been created
    try:
        assert dtf.shape[1] == num_cols + num_cat
        logging.info("All new columns created. SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper(). Incorrect number of columns created")
        raise err

    return dtf


def test_perform_feature_engineering(dtf):
    '''
    Test for perform_feature_engineering
    '''
    # Check the overall function for errors
    try:
        x_train, x_test, y_train, y_test = cls.perform_feature_engineering(dtf, [
        ])
        logging.info("Feature engineering ok. SUCCESS")
    except Exception as err:
        logging.error(
            "Testing perform_feature_engineering(). Some error ocurred.")
        raise err

    # Check dimensions
    try:
        assert x_train.shape[1] == x_test.shape[1]
        assert len(y_train.shape) == 1
        assert len(y_test.shape) == 1
        assert x_train.shape[0] == y_train.shape[0]
        assert x_test.shape[0] == y_test.shape[0]
        logging.info(
            "Dimensions of X and y for train and test are ok. SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing perform_frature_engineering(). Error in matrices dimensions. %s",
            err)
        raise err

    return x_train, x_test, y_train, y_test


def test_train_models(x_train, y_train):
    '''
    Test for train_model() function
    '''
    # Overall test
    try:
        cls.train_models(x_train, y_train)
        logging.info("Training done. SUCCESS")
    except Exception as err:
        logging.error("Testing train_models(). Something went wrong. %s", err)
        raise err

    # Check that model files are in their right place
    expected_files = ['logistic_model.pkl', 'rfc_model.pkl']
    models_path = r"./models/"
    try:
        for filename in expected_files:
            assert os.path.isfile(models_path + filename)
        logging.info("All expected model files are in the directory. SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing train_models(). Not all the expected models have been created. %s",
            err)
        raise err


def test_save_results(x_train, x_test, y_train, y_test):
    '''
    Tests for save_results() function
    '''
    # Overall test
    try:
        cls.save_results(x_train, x_test, y_train, y_test)
        logging.info("Results recorded properly")
    except Exception as err:
        logging.error("Testing save_results(). Something went wrong. %s", err)
        raise err

    # Check for expected files in expected directory
    expected_files = [
        'ROC_curves.png',
        'Tree_explainer.png',
        'Importances.png']
    try:
        for filename in expected_files:
            assert os.path.isfile(OUTPUT_PATH + filename)
        logging.info("All expected image files are in the directory. SUCCESS")
    except AssertionError as err:
        logging.warning(
            "Testing save_results(). Not all expected image files have been created. %s",
            err)


if __name__ == "__main__":
    # Get data to do the tests
    try:
        data = pd.read_csv(INPUT_PATH)
    except Exception as err:
        raise Exception(
            f"Error. Be sure the input file is available in path {INPUT_PATH}") from err

    # Test_import_data
    data = test_import_data(data)

    # Test perform EDA
    data = test_perform_eda(data)

    # Test encode the columns
    data = test_encoder_helper(data)

    # Test for feature engineering
    xtrain, xtest, ytrain, ytest = test_perform_feature_engineering(data)

    # Test training
    test_train_models(xtrain, ytrain)

    # Test record results to files
    test_save_results(xtrain, xtest, ytrain, ytest)
