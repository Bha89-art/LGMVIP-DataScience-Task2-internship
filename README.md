# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import numpy as np
import plotly.express as px
import shap

# Load the dataset
df = pd.read_csv('iris_dataset new corret.csv')

# Let's check out the first few rows to see what we're working with
print(df.head())

# Encoding the target variable
df['target'] = df['target'].astype('category').cat.codes

# Separating features and target
X = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
y = df['target']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Setting up grid search for hyperparameter tuning
param_grid = {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}
grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_clf = grid_search.best_estimator_

# Making predictions on the test set
y_pred = best_clf.predict(X_test)

# Evaluating the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Our model’s accuracy: {accuracy:.2f}')

# Displaying a detailed classification report
class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# Plotting the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Cross-validation scores
cv_scores = cross_val_score(best_clf, X, y, cv=5)
print(f'\nCross-Validation Scores: {cv_scores}')
print(f'Average Cross-Validation Score: {np.mean(cv_scores):.2f}')

# Feature importance visualization
plt.figure(figsize=(8, 6))
importances = best_clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

plt.title('Feature Importances')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), features[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()

# SHAP values for interpreting the model
explainer = shap.TreeExplainer(best_clf)
shap_values = explainer.shap_values(X_test)

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, feature_names=X.columns)
plt.show()

# Interactive feature importance with Plotly
fig = px.treemap(pd.DataFrame({'features': X.columns, 'importance': importances}), path=['features'], values='importance')
fig.update_layout(title='Interactive Feature Importance')
fig.show()

# Visualizing the decision tree with Matplotlib
plt.figure(figsize=(12, 8))
plot_tree(best_clf, filled=True, feature_names=X.columns, class_names=class_names)
plt.title('Decision Tree Visualization')
plt.show()
