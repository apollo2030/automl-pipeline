# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary

The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.

The classification goal is to predict if the client will subscribe to a term deposit with the bank.

## Scikit-learn Pipeline

The pipline consist of 5 big parts:
1. Training step which is defined in the `train.py` file and consist of a number of substeps: 
   - data aquisition in a pandas dataframe
   - cleaning and one hot encoding the data
   - splitting the data into the train and evaluation datasets
   - fitting a LogisticRegression model to the data (using the provided hyperparameters)
   - scoring the model
   - (optional) saving the model
2. HyperDrive run step used to find the best hyperparameters.
3. Selection of the best hyperdrive run
4. Training the model with the best parameters
5. Saving the model 

The training step uses Scikit-learn Logistic regression classifier.

### Logistic regression
Logistic regression, despite its name, is a linear model for classification rather than regression.
In this model, the probabilities describing the possible outcomes of a single trial are modeled using a logistic function. 

A logistic function or logistic curve is a common S-shaped curve (sigmoid curve).
In this classifier, separate binary classifiers are trained for all classes. This happens under the hood, so LogisticRegression instances using this solver behave as multiclass classifiers.

### Hyperdrive run
The idea of the HyperDrive run is to let HyperDrive find the best hyperparameters that will be later used to train the model.

The hyperparameters used in the training script are:
- --C which is the regularization strength and is a continuous value
- --max_itter is the max number iterations to converge and is a discrete value
- --save is a flag to let the script save the learned model.

In order to start a HyperDrive run we need to specify a **sampler**, an **estimator** and a **early stopping policy**. 

The sampler is a class that defines the behaviour of hyperdrive while selecting values for the parameter. 

In this case I have chosen the **RandomParameterSampling** because "Random sampling allows the search space to include both discrete and continuous hyperparameters." which is exactly what we have.

The regularization strength is a linear value that is why the sampling is `uniform` from 0.5 to .99.
the max_itter is discrete which fits the `choice` sampling strategy with incremental values from 1 to 128.

The benefits of using the **RandomParameterSampling** are that the number of runs can be controlled and it provides a way to pinpoint a range of parameters that can be refined further with a **GridParameterSampling**.

**RandomParameterSampling** is great for coarse guessing the right parameters, especially compared with **GridParameterSampling** which performs a grid search and has a P1xP2xP3 number of runs where P1, P2, P3 are the parameters to optimise. 

The **GridParameterSampling** doesn`t allow for continuous parameters.

The benefits of using a **GridParameterSampling** is that it allows for exploration of the entire parameter search space.

An estimator is a instance of the `SKLearn` estimator that actually runs the training script.

An early stopping policy is a configuration that specify the conditions to stop the run.
The stopping policy chosen is **Bandit** that offers a way to account for local minima `slack_factor` and a way to apply it after a specific number of runs `evaluation_interval` which makes this policy well ballanced in terms of flexibility.

`slack_factor` is a parameter of the Bandit policy that accounts for deacrease in the value of the primary metric but not enough to stop the run. For instance if the the accuracy of the last run is 0.7 and the accuracy of the current run is 0.6 with a slack_factor of 0.2 the policy will not stop the run.

`evaluation_interval` is the parameter that countrols the frequency of evaluation of the primary metric.

### Trainign the model with the best parameters
After the HyperDrive run we got a set of parameters that we can use to train the model.

### Saving the model
To save the model the third parameter `--save` is used. I have chosen to register the model in the azureml workspace to make it easier later to refer to it for deployment. 


## AutoML

Automl instead of choosing the LogisticRegression model has chosen the AssembleVoting algorithm with the following parameters: 

`"ensembled_iterations": "[1, 0, 19, 14, 11, 23, 6, 4, 5]"`

`"ensembled_algorithms": "['XGBoostClassifier', 'LightGBM', 'LightGBM', 'XGBoostClassifier', 'XGBoostClassifier', 'LightGBM', 'XGBoostClassifier', 'RandomForest', 'RandomForest']",`

`"ensemble_weights": "[0.21428571428571427, 0.14285714285714285, 0.21428571428571427, 0.07142857142857142, 0.07142857142857142, 0.07142857142857142, 0.07142857142857142, 0.07142857142857142, 0.07142857142857142]"`

The 2 pipelines are different because in the first case the algorithm was preselected based on the problem type. The automl pipeline is a bit simpler to setup and gives a better accuracy than the scikit-learn based pipeline. The accuracy for the scikit-learn pipeline is: 0.9089584850691915 compared to 0.91715 - the accuracy of the automl pipeline.

## Future work
In a subsequente experiment I would like to test if a different range for values in the hyperdrive parameter sampler could improve the score of the scikit-learn based solution.

One area of improvement would be setting multi_class to “multinomial” for the LogisticRegression with "sag" and "saga" solvers learns a true multinomial logistic regression model, which means that its probability estimates should be better calibrated than the default “one-vs-rest” setting.

Another thing to test would be changing the HyperDrive sampler from a RandomParameterSampler to a different sampler that could converge faster.
