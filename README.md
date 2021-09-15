# Task
It is necessary to build a sales forecast model for each store 31 days in advance. There are 20 stores (id).
Evaluation Metric - Scaled Mean Absolute Error (sMAE).

## Decision evaluation
Assessment of the decision will be made according to the following criteria:
- Achieved metric value: sMAE < 0.12 is desirable.
- Exploratory data analysis: what characteristics of the source data were analyzed, what conclusions and/or
hypotheses were put forward, what types of analysis were performed.
- Code quality: structured, understandable (clean), documented, computational optimality.
- The choice of machine learning method, its validity.
- The choice of a validation strategy, its validity.
- Selection of hyperparameters for the selected machine learning method (if any): what values, by which
strategy were selected.
- Conclusions from the results of the study

## Data files
- Training sample file: train.csv
- File with test sample (without target variables): test.csv
- Example file format with the result of the prediction: predict_example.csv

# Comments on the solution of the task
The solution is presented in three Jupyter notebooks in the folder named `notebooks`.
- `01_data_research.ipynb` - data research and analysis
- `02_model_building_ (id = 0) .ipynb` - building a model for store_id = 0 with detailed comments, visualizations and 
wrapping the process into a class (the class of model see in folder `models` too)
- `03_calculation_of_sMAE_and_test-predict, all stores.ipynb` - calculations of sMAE metrics and forecasts for all 
stores using previously built models and classes. There is the summary at the end of this notebook
- for each store was selected individual hyperparameters, see subfolder `parameters_selection` in the folder `notebooks`
- individual hyperparameters have been stored in the folder `model_settings`
- final predictions are presented in the `test_predict.csv` file in the folder named `data`