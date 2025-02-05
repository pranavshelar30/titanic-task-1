import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
df= pd.read_csv("/kaggle/input/titanic-task1-csv/Titanic-Dataset (2).csv")
df
df.shape
df.head()
df.describe()
df.duplicated().sum()
Survived = df['Survived'].value_counts().reset_index()
Survived
data = {'Survived': ['Male - No', 'Male - Yes', 'Female - No', 'Female - Yes'],
        'Counts': [100, 50, 30, 80]}  
Survived = pd.DataFrame(data)

plt.figure(figsize=(8, 6))
plt.bar(Survived['Survived'], Survived['Counts'],color=["Yellow","blue","red","Green"])
plt.xticks(Survived['Survived'])
plt.title('Comparison of Survival')
plt.xlabel('Gender and Survival Status')
plt.ylabel('Number of People')
plt.show()
df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)
df.head()
inputs = df.drop('Survived',axis='columns')
target = df['Survived']
counts = df.groupby(['Survived', 'Sex']).size().unstack().fillna(0)

bar_width = 0.35
index = counts.index


fig, ax = plt.subplots()


bar1 = ax.bar(index - bar_width/2, counts['male'], bar_width, label='male')
bar2 = ax.bar(index + bar_width/2, counts['female'], bar_width, label='female')


ax.set_xlabel('Survived')
ax.set_ylabel('Count')
ax.set_title('Survival Counts by Gender')
ax.set_xticks(index)
ax.set_xticklabels(['Not Survived', 'Survived'])
ax.legend()


plt.show()
X_train, X_test, y_train, y_test=train_test_split(inputs,target,test_size=0.1)
X_train
X_test
y_train
y_test


11     1
790    0
665    0
685    0
157    0
      ..
418    0
385    0
882    0
164    0
581    1
Name: Survived, Length: 90, dtype: int64
