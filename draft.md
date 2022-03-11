# Honest ML: A library for building confidence in statistical models

By Eric Schles

## Abstract

Machine learning metrics are built around the idea of training a model and then making out of sample predictions to test generalizability.  To do this there are a few standard methods - spliting the data into training and testing data and then predicting once on the testing or out of sample data.  Using cross validation to train on partitions of the data and then test on using one partition as the hold out, and averaging the metric across all partitions.  And finally there is stratified partitioning, which splits the data subject to some condition, usually on the proportion of labels in the entire dataset.  In this paper, we will look at a library that makes use of a different method, training the model on a very large number of train test splits and recording the out of sample error across these many splits.  By many we mean 500 to several thousand splits.  This creates a higher degree of confidence in the model and more closely simulates the likely scenarios you would find in the production setting, even with reasonably small datasets.

## Introduction

The idea of out of sample prediction is described in detail throughout the literature[1], the basic idea is to split the data into two groups, a training sample and a testing sample.  Once the data is split then a statistical model is trained on the training sample.  Then the trained model is used to predict the dependent variable from the independent variables[4] in the testing sample. Finally, a loss metric, like mean squared error[2] is used if it is a regression problem or cross entropy[3] is used if it's classification, to compare the predicted dependent variable against the ground truth dependent variable.  

This method can be useful as a first pass to assess model quality, however it has many deficiencies[5]#To Do add more references here#.  Since we only split the data once and we are dealing with a classification problem, we must hope for a few things:

1. We don't get a substantially different balance in the label classes in training and testing.  And that this balance is not different from the total data set, as well as, the population data in question.

2. We don't get a concentration of indepedent variables that are caused by a specific exogenous effect[6] in the training data and a different exogenous effect in the testing data.  

If either of these conditions fail then our loss metric may record either a far too optimistic or pessimistic view of how well the model does.  This in turn may have consequences for a whole host of things - failure to select the correct model, for instance, we may select a logistic regression model[7] when a decision tree model[8] is more approriate.  Or we may select the wrong hyperparameters for a given model class.  A direct consequence of a bad model is poor inference which may have difficult or impossible to recognize consequences, in some cases`[9][10][11][12]`.  Therefore it is of paramount importance that our models be 'honest' and the error well captured.  

To deal with this failure to generalize from a single training and testing split, cross validation[13] was created to increase the number of training and testing splits and then average the error metric or metrics.  The way this works is by creating a number of random partitions of the data, and then treating one of the partitions as out of sample, while the rest are treated as in sample.  Then the model is trained on all in sample predictions and the out of sample is left for predicting against, just like before.  The procedure is repeated for each partition, so that each partition is treated as both training and testing.  Finally the recorded metrics across each partition are averaged and reported, as well as the individual loss metrics.  The issue with this strategy is you need to tune the number of partitions - too many and individual partitions won't generalize well, too few and you will run into the same issues as with train test split once.  

In theory, both of the methods described are enough, the issue comes down to what happens in practice.  Therefore we have created [honest_ml](https://github.com/EricSchles/honest_ml) a library to do many individual splits of the data, typically on the order of 500 to several thousand.  The idea is to iterate over the random seed used in a typical train-test split implementation.  For this library, we use scikit-learn's implementation[14], consider the gold standard by many.  By doing so we remove the need to consider how many partitions is the right number.  Additionally, we far less likely to deal with a lucky or unlucky split, because we are splitting so many times.  

## Honest ML

Recap and expand on what honest ml does.  

## Experiments

## Discussion

## Conclusion



citation:

1 - [Montgomery, D.C., 1991. Nested or hierarchical designs. In: Design and Analysis of Experiments, third ed. John Wiley and Sons, New York, pp. 439–452](https://scholar.google.com/scholar?q=Montgomery,%20D.C.,%201991.%20Nested%20or%20hierarchical%20designs.%20In:%20Design%20and%20Analysis%20of%20Experiments,%20third%20ed.%20John%20Wiley%20and%20Sons,%20New%20York,%20pp.%20439452.)

2 - [Mean Squared Error Wikipedia](https://en.wikipedia.org/wiki/Mean_squared_error)

3 - [Cross Entropy Wikipedia](https://en.wikipedia.org/wiki/Cross_entropy)

4 - [Dependent and indepent variables](https://en.wikipedia.org/wiki/Dependent_and_independent_variables)

5 - [A critical look at the current train/test split in machine learning](https://arxiv.org/pdf/2106.04525.pdf)

6 - [Exogenous and Endogenous variables Wikipedia](https://en.wikipedia.org/wiki/Exogenous_and_endogenous_variables)

7 - [Logistic Regression Wikipedia](https://en.wikipedia.org/wiki/Logistic_regression)

8 - [Decision Tree Wikipedia](https://en.wikipedia.org/wiki/Decision_tree_learning)

9 - [Model Debugging Strategies Machine Learning](https://neptune.ai/blog/model-debugging-strategies-machine-learning)

10 - [Machine Learning - Testing and Debugging](https://developers.google.com/machine-learning/testing-debugging/common/overview)

11 - [Strategies for model debugging](https://towardsdatascience.com/strategies-for-model-debugging-aa822f1097ce)

12 - [The Ultimate Guide to Debugging your Machine Learning models](https://towardsdatascience.com/the-ultimate-guide-to-debugging-your-machine-learning-models-103dc0f9e421)

13 - [Cross Validation Wikipedia](https://en.wikipedia.org/wiki/Cross-validation_(statistics))

14 - [Sci-kit learn's train test split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)