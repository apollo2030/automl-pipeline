# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary

The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.

The classification goal is to predict if the client will subscribe to a term deposit with the bank.

## Scikit-learn Pipeline

The idea of the HyperDrive pipeline is to let HyperDrive find the best hyperparameters that will be later used to train the model.

Data for this run is cleaned and one hot encoded in the trainig script which has 3 parameters:
--C which is the regularization strength and is a continuous value
--max_itter is the max number iterations to converge and is a discrete value
--save is a flag to let the script save the learned model.

The RandomParameterSampling sammpler is based on the nature of the training script parameters. 
The regularization strength is a linear value that is why the sampling is uniform from 0.5 to .99.
the max_itter is discrete which fits the `choice` sampling strategy with incremental values from 1 to 128.

The stopping policy chosen is Bandit that offers a way to account for local minima `slack_factor` and a way to apply it after a specific number of runs `evaluation_interval` which makes this policy well ballanced in terms of flexibility.

## AutoML

Automl instead of choosing the LogisticRegression model has chosen the AssembleVoting algorithm with the following parameters: 

`"ensembled_iterations": "[1, 0, 19, 14, 11, 23, 6, 4, 5]",
"ensembled_algorithms": "['XGBoostClassifier', 'LightGBM', 'LightGBM', 'XGBoostClassifier', 'XGBoostClassifier', 'LightGBM', 'XGBoostClassifier', 'RandomForest', 'RandomForest']",
"ensemble_weights": "[0.21428571428571427, 0.14285714285714285, 0.21428571428571427, 0.07142857142857142, 0.07142857142857142, 0.07142857142857142, 0.07142857142857142, 0.07142857142857142, 0.07142857142857142]"`

The 2 pipelines are different because in the first case the algorithm was preselected based on the problem type. The automl pipeline is a bit simpler to setup and gives a better accuracy than the scikit-learn based pipeline. The accuracy for the scikit-learn pipeline is: 0.9089584850691915 compared to 0.91715 - the accuracy of the automl pipeline.

## Future work
In a subsequente experiment I would like to test if a different range for values in the hyperdrive parameter sampler could improve the score of the scikit-learn based solution.
