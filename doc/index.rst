.. breton_cretenet documentation master file, created by
   sphinx-quickstart on Wed Mar 29 14:47:26 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to breton_cretenet's documentation!
===========================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

Main results
============
In the table bellow, you can find the test and training score (Mean Absolute Error) of our linear regression model.
The task was to predict the price of a house based on several features. We do not provide more information about the dataset since this is a well-known dataset that we use for a toy project.
To use other models or perform other types of tasks on other datasets, please type ``breton_cretenet_results --help``.
As you can see in the table, each model is trained and tested with 3 different random states.

+----------------------+----------------------+----------------------+
|                      |     Random state     |        linear        |
+======================+======================+======================+
| model                |          42          |  LinearRegression()  |
+----------------------+----------------------+----------------------+
| score_test           |          42          |   4.24265375592562   |
+----------------------+----------------------+----------------------+
| score_train          |          42          |   2.269868774064942  |
+----------------------+----------------------+----------------------+
| model                |          84          |  LinearRegression()  |
+----------------------+----------------------+----------------------+
| score_test           |          84          |   4.835212745285229  |
+----------------------+----------------------+----------------------+
| score_train          |          84          |  1.8993063048667522  |
+----------------------+----------------------+----------------------+
| model                |          126         |  LinearRegression()  |
+----------------------+----------------------+----------------------+
| score_test           |          126         |   4.185290166938272  |
+----------------------+----------------------+----------------------+
| score_train          |          126         |  1.6031351348613696  |
+----------------------+----------------------+----------------------+

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
