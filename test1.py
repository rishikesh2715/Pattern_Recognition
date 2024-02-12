"""
Pattern Recogntiton Project 1
Rishikesh
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.ion()

# load the dataset
df = pd.read_excel('Proj1DataSet.xlsx')
df.rename(columns={
    'meas_1': 'SepL',
    'meas_2': 'SepW',
    'meas_3': 'PetL',
    'meas_4': 'PetW'
}, inplace=True)

# feature names
features = ['SepL', 'SepW', 'PetL', 'PetW']

"""
Statistics
"""

# loop through each feature to find min, max, mean and variance 
for feature in features:
    print(f"Statistics for {feature}:")
    print(f"Minimum: {np.min(df[feature])}")
    print(f"Maximum: {np.max(df[feature])}")
    print(f"Mean: {np.mean(df[feature])}")
    print(f"Variance: {np.var(df[feature])}\n")

# plotting Sepal Length' vs 'Sepal Width
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='SepL', y='SepW', hue='species', style='species', palette='Set1')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Scatter plot of Sepal Length vs. Sepal Width by Species')
plt.legend(title='Species')
plt.show()


# plotting Pedal Length vs Pedal Width
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='PetL', y='PetW', hue='species', style='species', palette='Set1')
plt.xlabel('Pedal Length')
plt.ylabel('Pedal Width')
plt.title('Scatter plot of Pedal Length vs. Pedal Width by Species')
plt.legend(title='Species')
plt.show()


# class probabilities
class_probs = df['species'].value_counts(normalize=True)

within_class_variances = {feature: 0 for feature in features}
between_class_variances = {feature: 0 for feature in features}

# within-class variance 
for species in df['species'].unique():
    species_data = df[df['species'] == species]
    for feature in features:
        within_class_variances[feature] += class_probs[species] * np.var(species_data[feature], ddof=1)

# mean of each feature in each class
feature_means_by_class = df.groupby('species')[features].mean()

# overall mean of each feature
overall_feature_means = df[features].mean()

# between-class variance 
for feature in features:
    between_class_variances[feature] = sum(class_probs[species] * (feature_means_by_class.at[species, feature] - overall_feature_means[feature]) ** 2 for species in df['species'].unique())


print("Within-Class Variances:", within_class_variances)
print("Between-Class Variances:", between_class_variances)
print()


"""
correlation coefficients
"""
species_numeric = df['species'].astype('category').cat.codes
df_for_corr = df[features].copy()
df_for_corr['Class'] = species_numeric

# correlation matrix
corrcoef = np.corrcoef(df_for_corr.T)

# plot the correlation matrix
plt.figure(figsize=(8, 6))
plt.imshow(corrcoef, cmap='jet', aspect='auto')
plt.colorbar()
plt.xticks(range(len(df_for_corr.columns)), df_for_corr.columns, rotation=90)
plt.yticks(range(len(df_for_corr.columns)), df_for_corr.columns)
plt.show()

"""
Four Features vs the class label
"""

# plotting the four features vs the class labels
plt.figure(figsize=(8, 6))

# create a list of titles
titles = ['SepL vs Class', 'SepW vs Class', 'PetL vs Class', 'PetW vs Class']

# Loop through each feature and create a subplot for each
for i, feature in enumerate(features):
    plt.subplot(2, 2, i+1)  # 2x2 grid, position i+1
    plt.scatter(df[feature], species_numeric + 1, marker='x', c='red')
    plt.title(titles[i])
    plt.xlabel(feature)
    plt.ylabel('Class')
    plt.xlim(0, 8)
    plt.ylim(1, 3)

plt.tight_layout()
plt.show()




"""
Classification
Batch_perceptron and LS
Setosa Vs Versi+Virigi
All features 
"""
print()
print('Setosa Vs. Versi + Virgi, All Features')

df['Class'] = np.where(df['species'] == 'setosa', 1, -1)

X = df[features].values
X = np.hstack((X, np.ones((X.shape[0], 1))))  # add a column of 1's for the bias term
y = df['Class'].values.reshape(-1, 1)

# initialize with small random weights
w_p = np.random.rand(5, 1)
# print(w_p)

# batch Perceptron 
rho = 0.001  # learning rate
epochs = 0
max_epochs = 5000
while True:
    misclassified = 0
    for i in range(len(X)):
        if y[i] * (np.dot(w_p.T, X[i])) <= 0:
            w_p += rho * y[i] * X[i].reshape(-1, 1)
            misclassified += 1
    epochs += 1
    if misclassified == 0 or epochs > max_epochs:
        break

if misclassified == 0:
    print("The Batch Perceptron converged after", epochs, "epochs")
else:
    print("The Batch Perceptron did not converge after", max_epochs, "epochs")

print('Batch Perceptron - Weights:', w_p)

misclassified_perceptron = 0
for i in range(len(X)):
    if y[i] * (np.dot(w_p.T, X[i])) <= 0:
        misclassified_perceptron += 1
print('Batch Perceptron - Misclassifications:', misclassified_perceptron)

# least Squares 
X_T = X.T
X_T_X = X_T.dot(X)
X_T_X_inv = np.linalg.inv(X_T_X)
X_pseudo_inv = X_T_X_inv.dot(X_T)
w_ls = X_pseudo_inv.dot(y)

print('Least Squares - Weights):', w_ls)

misclassified_ls = 0
for i in range(len(X)):
    if np.sign(np.dot(w_ls.T, X[i])) != y[i]:
        misclassified_ls += 1
print('Least Squares - Misclassifications:', misclassified_ls)




"""
Classification
Batch_perceptron and LS
Setosa Vs. Versi+Virigi
Features 3 and 4 Only
"""
print()
print('Setosa Vs. Versi + Virgi, Features 3 and 4')
df['Class'] = np.where(df['species'] == 'setosa', 1, -1)

X = df[['PetL', 'PetW']].values
X = np.hstack((X, np.ones((X.shape[0], 1))))  # add a column of 1's for the bias term
y = df['Class'].values.reshape(-1, 1)

# batch Perceptron
w_p = np.random.rand(3, 1)
rho = 0.001  # learning rate
epochs = 0
max_ephocs = 5000
while True:
    misclassified = 0
    for i in range(len(X)):
        if y[i] * (np.dot(w_p.T, X[i])) <= 0:
            w_p += rho * y[i] * X[i].reshape(-1, 1)
            misclassified += 1
    epochs += 1
    if misclassified == 0 or epochs >= max_ephocs:
        break

if misclassified == 0:
    print("The Batch Perceptron converged after", epochs, "epochs")
else:
    print("The Batch Perceptron did not converge after", max_epochs, "epochs")

print('Batch Perceptron - Weights:', w_p)

misclassified_perceptron = 0
for i in range(len(X)):
    if y[i] * (np.dot(w_p.T, X[i])) <= 0:
        misclassified_perceptron += 1
print('Batch Perceptron - Misclassifications:', misclassified_perceptron)

# least Squares
X_T = X.T
X_T_X = X_T.dot(X)
X_T_X_inv = np.linalg.inv(X_T_X)
X_pseudo_inv = X_T_X_inv.dot(X_T)

w_ls = X_pseudo_inv.dot(y)

print('Least Squares - Weights:', w_ls)

misclassified_ls = 0
for i in range(len(X)):
    if np.sign(np.dot(w_ls.T, X[i])) != y[i]:
        misclassified_ls += 1
print('Least Squares - Misclassifications:', misclassified_ls)


# plotting decision boundary and features vectors
x_values = np.linspace(0, 7, 100)
y_values_perceptron = -(w_p[0] * x_values + w_p[2]) / w_p[1]
y_values_ls = -(w_ls[0] * x_values + w_ls[2]) / w_ls[1]

plt.figure(figsize=(8, 6))
plt.scatter(df.loc[df['species'] == 'setosa', 'PetL'], df.loc[df['species'] == 'setosa', 'PetW'], c='red', label='Setosa')
plt.scatter(df.loc[df['species'] != 'setosa', 'PetL'], df.loc[df['species'] != 'setosa', 'PetW'], c='blue', label='Versi+Virigi', marker='x')
plt.plot(x_values, y_values_perceptron, label='Batch Perceptron Decision Boundary', color='green')
plt.plot(x_values, y_values_ls, label='Least Squares Decision Boundary', color='orange')
plt.xlim(0.5, 7)
plt.ylim(-0.5, 3)

# labels, legneds and title
plt.xlabel('Pedal Length')
plt.ylabel('Pedal Width')
plt.title('Setosa Vs. Versi+Virigi, Features 3 and 4')
plt.legend()
plt.show()





"""
Classification
Batch_perceptron and LS
Virgi Vs. Versi+Setosa
All Features
"""

print()
print('Virgi Vs. Versi + Setosa, All Features')
df['Class'] = np.where(df['species'] == 'virginica', 1, -1)


X = df[features].values
X = np.hstack((X, np.ones((X.shape[0], 1))))  # add a column of ones for the bias term
y = df['Class'].values.reshape(-1, 1)

# initialize with small random weights
w_p = np.random.rand(5, 1) 

# batch Perceptron
rho = 0.001  # learning rate
max_epochs = 5000
epochs = 0
while True:
    misclassified = 0
    for i in range(len(X)):
        if y[i] * (np.dot(w_p.T, X[i])) <= 0:
            w_p += rho * y[i] * X[i].reshape(-1, 1)
            misclassified += 1
    epochs += 1
    if misclassified == 0 or epochs >= max_epochs:
        break

if misclassified == 0:
    print("The Perceptron has converged after", epochs, "epochs.")
else:
    print("The Batch Perceptron did not converge after", max_epochs, "epochs")

print('Batch Perceptron - Weights:', w_p)

misclassified_perceptron = 0
for i in range(len(X)):
    if y[i] * (np.dot(w_p.T, X[i])) <= 0:
        misclassified_perceptron += 1
print('Batch Perceptron - Misclassifications:', misclassified_perceptron)

# least Squares Classification
X_T = X.T
X_T_X = X_T.dot(X)
X_T_X_inv = np.linalg.inv(X_T_X)
X_pseudo_inv = X_T_X_inv.dot(X_T)

w_ls = X_pseudo_inv.dot(y)

print('Least Squares - Weights:', w_ls)

misclassified_ls = 0
for i in range(len(X)):
    if np.sign(np.dot(w_ls.T, X[i])) != y[i]:
        misclassified_ls += 1
print('Least Squares - Misclassifications:', misclassified_ls)




"""
Classification
Batch_perceptron and LS
Virgi Vs. Versi+Setosa
Features 3 and 4 Only 
"""
print()
print('Virgi Vs. Versi + Setosa, Features 3 and 4')

df['Class'] = np.where(df['species'] == 'virginica', 1, -1)
X = df[['PetL', 'PetW']].values
X = np.hstack((X, np.ones((X.shape[0], 1))))  # add a column of ones for the bias term
y = df['Class'].values.reshape(-1, 1)

# initialize with small random weights
w_p = np.random.rand(3, 1) 

# batch Perceptron
rho = 0.001  # learning rate
epochs = 0
max_epochs = 5000
while True:
    misclassified = 0
    for i in range(len(X)):
        if y[i] * (np.dot(w_p.T, X[i])) <= 0:
            w_p += rho * y[i] * X[i].reshape(-1, 1)
            misclassified += 1
    epochs += 1
    if misclassified == 0 or epochs >= max_epochs:
        break

if misclassified == 0:
    print("The Perceptron has converged after", epochs, "epochs.")
else:
    print("The Batch Perceptron did not converge after", max_epochs, "epochs")

print('Batch Perceptron - Weights:', w_p)

misclassified_perceptron = 0
for i in range(len(X)):
    if y[i] * (np.dot(w_p.T, X[i])) <= 0:
        misclassified_perceptron += 1
print('Batch Perceptron - Misclassifications:', misclassified_perceptron)

# least Squares Classification
X_T = X.T
X_T_X = X_T.dot(X)
X_T_X_inv = np.linalg.inv(X_T_X)
X_pseudo_inv = X_T_X_inv.dot(X_T)

w_ls = X_pseudo_inv.dot(y)

print('Least Squares - Weights:', w_ls)

misclassified_ls = 0
for i in range(len(X)):
    if np.sign(np.dot(w_ls.T, X[i])) != y[i]:
        misclassified_ls += 1
print('Least Squares - Misclassifications:', misclassified_ls)


x_values = np.linspace(0, 7, 100)

y_values_perceptron = -(w_p[0] * x_values + w_p[2]) / w_p[1]
y_values_ls = -(w_ls[0] * x_values + w_ls[2]) / w_ls[1]

# plotting decision boundary and features vectors
plt.figure(figsize=(8, 6))
# plt.scatter(df.loc[df['species'] == 'setosa', 'PetL'], df.loc[df['species'] == 'setosa', 'PetW'], c='red', label='Setosa', marker='x')
plt.scatter(df.loc[df['species'] == 'virginica', 'PetL'], df.loc[df['species'] == 'virginica', 'PetW'], c='blue', label='Virginica', marker = 'x')
plt.scatter(df.loc[df['species'] != 'virginica', 'PetL'], df.loc[df['species'] != 'virginica', 'PetW'], c='red', label='Versi+Setosa')

plt.plot(x_values, y_values_perceptron, label='Batch Perceptron Decision Boundary', color='green')
plt.plot(x_values, y_values_ls, label='Least Squares Decision Boundary', color='orange')
plt.xlim(0.5, 7)
plt.ylim(-0.5, 3)

# labels, legend, and title
plt.xlabel('Pedal Length')
plt.ylabel('Pedal Width')
plt.title('Virginica Vs. Versi+Setosa, Features 3 and 4')
plt.legend()
plt.show()




"""
Classification
Multiclass LS
Setosa Vs. Versi Vs. Virigi
Features 3 and 4 Only 
"""
print()
print('Setosa Vs. Versi Vs. Virgi, Features 3 and 4')

X = df[['PetL', 'PetW']].values
X_augmented = np.hstack((X, np.ones((X.shape[0], 1))))  # add a column of 1's for the bias term

# onehot encoding
t = np.zeros((df.shape[0], 3))
t[df['species'] == 'setosa', 0] = 1
t[df['species'] == 'versicolor', 1] = 1
t[df['species'] == 'virginica', 2] = 1

# least squares classification
X_T = X_augmented.T
X_T_X = X_T.dot(X_augmented)
X_T_X_inv = np.linalg.inv(X_T_X)
X_pseudo_inv = X_T_X_inv.dot(X_T)

W_ls = X_pseudo_inv.dot(t)

print('Least Squares - Weights:', W_ls)

# calculating the misclassifications
predictions = X_augmented.dot(W_ls)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(t, axis=1)
misclassified = np.sum(predicted_classes != true_classes)
print('Misclassifications: ', misclassified)



# create a meshgrid
xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200),
                     np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200))

grid = np.c_[xx.ravel(), yy.ravel(), np.ones(xx.ravel().shape[0])]
scores = grid.dot(W_ls)
predicted_classes = np.argmax(scores, axis=1).reshape(xx.shape)

# plot the features
plt.figure(figsize=(8, 6))
plt.scatter(df[df['species'] == 'setosa']['PetL'], df[df['species'] == 'setosa']['PetW'], label='Setosa', marker='o')
plt.scatter(df[df['species'] == 'versicolor']['PetL'], df[df['species'] == 'versicolor']['PetW'], label='Versicolor', marker='+')
plt.scatter(df[df['species'] == 'virginica']['PetL'], df[df['species'] == 'virginica']['PetW'], label='Virginica', marker='x')

# the line where class 0 score equals class 1 score
w_diff_01 = W_ls[:, 0] - W_ls[:, 1]
y_values_01 = -(w_diff_01[0] * x_values + w_diff_01[2]) / w_diff_01[1]

# the line where class 1 score equals class 2 score
w_diff_12 = W_ls[:, 1] - W_ls[:, 2]
y_values_12 = -(w_diff_12[0] * x_values + w_diff_12[2]) / w_diff_12[1]

# the line where class 0 score equals class 2 score
w_diff_02 = W_ls[:, 0] - W_ls[:, 2]
y_values_02 = -(w_diff_02[0] * x_values + w_diff_02[2]) / w_diff_02[1]

# plotting the decision boundaries
plt.plot(x_values, y_values_01, label='Decision Boundary: Setosa vs Versicolor')
plt.plot(x_values, y_values_12, label='Decision Boundary: Versicolor vs Virginica')
plt.plot(x_values, y_values_02, label='Decision Boundary: Setosa vs Virginica')

plt.xlim(0.5, 7)
plt.ylim(-0.5, 3)


# labels, legends and title
plt.xlabel('Pedal Length')
plt.ylabel('Pedal Width')
plt.title('Multiclass Classification with LS: Setosa vs Versi vs Virigi')
plt.legend()
plt.show()

input("press a key to close script")


