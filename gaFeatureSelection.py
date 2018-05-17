import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from deap import creator, base, tools, algorithms

def avg(l):
    return (sum(l)/float(len(l)))

def getFitness(individual, X, y):
    # get index with value 0
    cols = [index for index in range(len(individual)) if individual[index] == 0]

    # get features subset
    X_parsed = X.drop(X.columns[cols], axis=1)
    X_subset = pd.get_dummies(X_parsed)

    # apply classification algorithm
    clf = LogisticRegression()

    return (avg(cross_val_score(clf, X_subset, y, cv=5)),)

def geneticAlgorithm(X, y):
    # create individual
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # create toolbox
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat,
        creator.Individual, toolbox.attr_bool, len(X.columns))
    toolbox.register("population", tools.initRepeat, list,
        toolbox.individual)
    toolbox.register("evaluate", getFitness, X=X, y=y)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # initialize parameters
    n_population = 50
    n_generation = 10
    pop = toolbox.population(n=n_population)
    hof = tools.HallOfFame(n_population * n_generation)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # genetic algorithm
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2,
        ngen=n_generation, stats=stats, halloffame=hof,
        verbose=True)

    # return hall of fame
    return hof

if __name__ == '__main__':

    # read dataframe from csv
    df = pd.read_csv('/home/renato/Documents/base_mama/csv_result-4classesFrontais.csv', sep=',')

    # encode labels column to numbers
    le = LabelEncoder()
    le.fit(df.iloc[:,-1])
    y = le.transform(df.iloc[:,-1])
    X = df.iloc[:,:-1]

    # get accuracy with all features
    individual = [1 for i in range(len(X.columns))]
    print("Accuracy with all features: \t" + str(getFitness(individual, X, y)) + "\n")

    # apply genetic algorithm
    hof = geneticAlgorithm(X,y)
