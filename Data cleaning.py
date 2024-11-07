import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

train_df.head()

train_df.info()

train_df.isnull().sum()

train_df.describe()


train_df['Age'].fillna(train_df['Age'].median(), inplace=True)


train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)


train_df.drop(columns=['Cabin'], inplace=True)


train_df['Title'] = train_df['Name'].str.extract(' ([A-Za-z]+).', expand=False)


train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1


sns.barplot(x='Sex', y='Survived', data=train_df)
plt.title('Survival Rate by Gender')
plt.show()


sns.barplot(x='Class', y='Survived', data=train_df)
plt.title('Survival Rate by Class')
plt.show()


plt.figure(figsize=(10, 6))
sns.histplot(train_df['Age'], kde=True, bins=30)
plt.title('Age Distribution')
plt.show()


plt.figure(figsize=(10, 6))
sns.barplot(x='Survived', y='Age', data=train_df)
plt.title('Age and Survival')
plt.show()


plt.figure(figsize=(10, 6))
sns.heatmap(train_df.corr(), annot=True, cmap='cool warm')
plt.title('Correlation Matrix')
plt.show()


train_df = pd.get_dummies(train_df, columns=['Sex', 'Embarked', 'Title'], drop_first=True)


X = train_df.drop(columns=['Name', 'Ticket', 'PassengerId', 'Survived'],errors='ignore')
y = train_df['Survived']

