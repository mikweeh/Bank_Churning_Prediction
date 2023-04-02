# Predict Customer Churn

## Project Description
The project consists of the application of the best programming practices from the point of view of joint programming with a team. Thus the project follows coding (PEP8) and engineering best practices for implementing software (modular, documented, and tested).

The project challenge was published on Kaggle and can be found [here](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers/code). It consists of identifying the banking service customers who are going to get churned based on a series of input variables.

## Files and data description
In the directory you will find three files:

* This file, a README.
* A python script called `churn_library.py`. It is the module that contain the helper functions to be imported.
* A python script called `churn_script_logging_and_test.py`. It contains all the necessary tests to check that the functions from `churn_library.py`work properly.

In addition you will find a folder with a .csv file called "_bank_data.csv_", This is the input data.

## Running Files

Both scripts can be run from the terminal to check they're working.

The `churn_library.py` contains a `__main__` block with a the full EDA. You just type in the terminal the command below to run it:

`$ python churn_library.py`

Please, check that you have in your project directory a folder called "_data_" with the input file called "_bank_data.csv_". This script creates in your project folder two new directories as long as they don't already exist: `output` and `models`

In the first, the images created from the DEA will be left. In the second one the ML models will be saved.

The `churn_script_logging_and_test.py` file also contains a `__main__` block. Since some function inputs depend on the outputs of previous functions the most recommended is run the full set of tests with the command:

`$ python churn_script_logging_and_test.py`

However, if you want to run all the tests without training again, you just have to comment the apropriate line in the main block, as long as a previous training had already been carried out and the models had been saved in the corresponding project folder.

This script creates a new folder called `logs` if it doesn't exist. In this folder a log file is created each time the script is run. The log file has the name _churn_library.log_

