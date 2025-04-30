# Linear Regression Implementation

This repository contains a Python implementation of linear regression with gradient descent optimization and various feature selection techniques.

## Features

- Data preprocessing and normalization
- Simple and multivariate linear regression
- Gradient descent optimization with customizable learning rates
- Analytical solution using pseudoinverse
- Forward feature selection algorithm
- Polynomial feature transformation

## Usage

```python
# Preprocessing and normalization
X_normalized, y_normalized = preprocess(X, y)

# Apply bias trick (add intercept term)
X_with_bias = apply_bias_trick(X_normalized)

# Solve using gradient descent
theta, loss_history = gradient_descent(X_with_bias, y_normalized, 
                                     init_theta, learning_rate, iterations)

# Solve analytically using pseudoinverse
theta_pinv = compute_pinv(X_with_bias, y_normalized)

# Find optimal learning rate
best_lr = find_best_learning_rate(X_train, y_train, X_val, y_val, iterations)

# Run gradient descent with stopping condition
theta, loss_history = gradient_descent_stop_condition(X_with_bias, y_normalized, 
                                                    init_theta, learning_rate, max_iterations)

# Feature selection
best_features = forward_feature_selection(X_train, y_train, X_val, y_val, 
                                        learning_rate, iterations)

# Create polynomial features
X_poly = create_square_features(df)
```

## Implemented Methods

- `preprocess(X, y)`: Standardize features and target variables
- `apply_bias_trick(X)`: Add intercept term to feature matrix
- `compute_loss(X, y, theta)`: Calculate mean squared error loss
- `gradient_descent(X, y, theta, eta, num_iters)`: Optimize parameters with gradient descent
- `compute_pinv(X, y)`: Calculate optimal parameters analytically
- `find_best_learning_rate(X_train, y_train, X_val, y_val, iterations)`: Find optimal learning rate
- `gradient_descent_stop_condition(X, y, theta, eta, max_iters)`: Gradient descent with convergence check
- `forward_feature_selection(X_train, y_train, X_val, y_val, eta, iterations)`: Select best features iteratively
- `create_square_features(df)`: Generate polynomial features of degree 2

## Dependencies

- NumPy
- Pandas
- Matplotlib (for visualization only)
