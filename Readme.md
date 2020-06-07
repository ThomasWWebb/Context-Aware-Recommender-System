# CARS - InCarMusic

A context aware recommender system that uses a user-based collabrative approach. A single context is used with the ratings from the InCarMusic dataset to provide music recommendations for the stored users 

## Use

There are three implementations, one that acts as a baseline with no context taken into account and two which each use a different single context; Driving Style and Mood. Running the python file of each CARS displays a simple UI that can be used to produce ratings for each stored user. All three can be evaluated with MAE, Recall and Precision using k-fold cross validation.   