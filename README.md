### Description
This is a project attempts and aims to realize machine learning algorithms by pure Python, including Decision Tree Models and basic Neuron Networks with Back Propagation. 

## Data
Datasets are usually manipulated with modules such as Pandas, here a class Data is defined as a list of Items (this should have been translated as Entries), each is defined as a dict in Python.
This realization of Data is surely insensible but simple at the first times, by predefining data structures and transposing the whole list of dicts as dict of lists can reduce its memory usage and improve its performance.
A preset dataset of "Melons" were translated from the book: 《Machine Learning》 by Zhihua Zhou (Published by Tsinghua University Press) and implemented in the files.

## Decision Tree
Decision tree models use a variance main_feature to specify which feature shall be involved in data gain (loss function of decision trees) and targeted in further optimization.
The basic optimization tactic follows the one algorithm mentioned in the book mentioned above.

## Neural Networks
Neural Networks are defined based on the structure of Neurons and Layers. Usually we use layers to specify a group of identical neurons. Although unnecessary, for the sake of flexibility of the network, layers can include a diversity of neurons here.
The networks are defined as multiple Layers connected to each other. Gradients as dictionary and Derivatives of the Activation Functions are predefined and inferred by the structure of the network. Back propagation is implemented and an example of training by SGD is given.
Here 2 types of neurons were successfully introduced: BP (Actually Linear + Any Activation), RBF (Radial Based Function, Gaussian Activation with adjustable parameters). RNN and LSTM were introduced but the author quited due to several reasons, mainly time schedules, thus they weren't fully realized.
