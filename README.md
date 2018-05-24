
# Genetic Algorithm For Feature Selection
> Feature selection is the process of finding the most relevant variables for a predictive model. These techniques can be used to identify and remove unneeded, irrelevant and redundant features that do not contribute or decrease the accuracy of the predictive model.

## Description
In nature, the genes of organisms tend to evolve over successive generations to better adapt to the environment. The Genetic Algorithm is an heuristic optimization method inspired by that procedures of natural evolution.

In feature selection, the function to optimize is the generalization performance of a predictive model. More specifically, we want to minimize the error of the model on an independent data set not used to create the model.



[font](https://www.neuraldesigner.com/blog/genetic_algorithms_for_feature_selection)




## Dependencies
[Pandas](https://pandas.pydata.org/)

[Numpy](http://www.numpy.org/)

[scikit-learn](http://scikit-learn.org/stable/)

[Deap](https://deap.readthedocs.io/en/master/)


## Usage example
1. Open the grSim.
1. Turn off all robots.
1. Put them all out of bounds.
1. Get one of the blue team robots, put it inside the field and turn on.
![](prints/exampleSimulation.png)

1. Get the Vision multicast adress, Vision multicast port and Command listen port on grSim.
![](prints/ips.png)

1. Go to `/ssl-client/ssl-Client/net/robocup_ssl_client.h` and paste the Vision Multicast adress and the Vision Multicast port on `string net_ref_address `and `int port`, respectively.
![](prints/clientH.png)

1. Go to `/ssl-client/ssl-Client/net/grSim_client.cpp` and paste the Vision Multicast adress and the Command listen port on `this->_addr.setAddress()`and `this->_port = quint16()`, respectively.
![](prints/myudpCPP.png)

1. Run!

#### Author: [Renato Sousa](https://github.com/renatoosousa)
