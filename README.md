
# Genetic Algorithm For Feature Selection
> Search the best feature subset for you classification model

## Description
Feature selection is the process of finding the most relevant variables for a predictive model. These techniques can be used to identify and remove unneeded, irrelevant and redundant features that do not contribute or decrease the accuracy of the predictive model.

In nature, the genes of organisms tend to evolve over successive generations to better adapt to the environment. The Genetic Algorithm is an heuristic optimization method inspired by that procedures of natural evolution.

In feature selection, the function to optimize is the generalization performance of a predictive model. More specifically, we want to minimize the error of the model on an independent data set not used to create the model.




## Dependencies
[Pandas](https://pandas.pydata.org/)

[Numpy](http://www.numpy.org/)

[scikit-learn](http://scikit-learn.org/stable/)

[Deap](https://deap.readthedocs.io/en/master/)


## Usage
1. Go to the repository folder
1. Run:
```
python gaFeatureSelection.py path n_population n_generation
```
Obs:
  - `path` should be the path to some dataframe in csv format
  - `n_population` and `n_generation` must be integers
  - You can go to the code and change the classifier so that the search is optimized for your classifier.

## Usage Example
```
python gaFeatureSelection.py datasets/nuclear.csv 20 6
```

## Fonts
1. This repository was heavily based on [GeneticAlgorithmFeatureSelection](https://github.com/scoliann/GeneticAlgorithmFeatureSelection)
1. For the description was used part of the introduction of  [Genetic algorithms for feature selection in Data Analytics](https://www.neuraldesigner.com/blog/genetic_algorithms_for_feature_selection). Great text.

#### Author: [Renato Sousa](https://github.com/renatoosousa)
