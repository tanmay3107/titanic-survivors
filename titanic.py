import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')


train_df['Age'].fillna(train_df['Age'].mean(), inplace=True)  
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)  
train_df.drop(columns=['Cabin', 'Ticket', 'Name'], inplace=True) 

train_df = pd.get_dummies(train_df, columns=['Sex', 'Embarked'], drop_first=True)

X = train_df.drop(columns=['Survived', 'PassengerId'])
y = train_df['Survived']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(
    random_state=42,
    max_depth=5,  
    min_samples_split=10,  
    min_samples_leaf=5  
)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)
print(f'Validation Accuracy: {accuracy * 100:.2f}%')

print(classification_report(y_val, y_pred))

cm = confusion_matrix(y_val, y_pred)
print('Confusion Matrix:')
print(cm)

plt.figure(figsize=(20,10))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=['Not Survived', 'Survived'], rounded=True)
plt.show()

test_df['Age'].fillna(test_df['Age'].mean(), inplace=True)  
test_df['Fare'].fillna(test_df['Fare'].mean(), inplace=True)  
test_df.drop(columns=['Cabin', 'Ticket', 'Name'], inplace=True)  

test_df = pd.get_dummies(test_df, columns=['Sex', 'Embarked'], drop_first=True)

X_test = test_df.drop(columns=['PassengerId'])

test_predictions = clf.predict(X_test)

submission_df = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': test_predictions
})

submission_df.to_csv('titanic_submission.csv', index=False)

print("Predictions saved to 'titanic_submission.csv'")
