# Decision Tree Classifier

A Python implementation of a decision tree classifier with support for different impurity measures and tree pruning techniques.

## Features

- Custom implementation of decision tree algorithm
- Multiple impurity measures (Gini impurity and Entropy)
- Information gain and gain ratio splitting criteria
- Tree pruning methods:
  - Chi-square statistical pruning
  - Max depth limitation
- Feature importance calculation

## Usage

```python
# Initialize a decision tree with Gini impurity
tree_gini = DecisionTree(data=X_train, impurity_func=calc_gini)
tree_gini.build_tree()

# Initialize a decision tree with Entropy and gain ratio
tree_entropy = DecisionTree(data=X_train, impurity_func=calc_entropy, gain_ratio=True)
tree_entropy.build_tree()

# Evaluate accuracy
accuracy = tree_gini.calc_accuracy(X_validation)
```

## Implemented Methods

- `calc_gini(data)`: Calculate Gini impurity
- `calc_entropy(data)`: Calculate entropy
- `goodness_of_split(feature)`: Calculate information gain or gain ratio
- `depth_pruning(X_train, X_validation)`: Evaluate different depth limits
- `chi_pruning(X_train, X_test)`: Pruning with chi-square test
- `predict(instance)`: Make predictions for new data

## Dependencies

- NumPy
- Matplotlib (for visualization only)
