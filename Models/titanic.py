import matplotlib.pyplot as plt
# matplotlib inline
import numpy as np
import pandas as pd
# import statsmodels.api as sm
# from statsmodels.nonparametric.kde import KDEUnivariate

# from statsmodels.nonparametric import smoothers_lowess
from pandas import Series, DataFrame
# from patsy import dmatrices
from sklearn import datasets, svm



df = pd.read_csv('/Users/arjun/Documents/Titanic/train.csv', header=0)


 #  To see the age ,sex and ticket columns
df[['Sex','Age','Ticket']]

#  Observing the values which are greater than 60
df[df['Age'] > 60][['Sex', 'Pclass', 'Age', 'Survived']]

# Printing the gae values that come across as null
df[df['Age'].isnull()][['Sex', 'Pclass', 'Age']]

#  Plotting the number of survived.
plt.figure(figsize=(6,4))
# fig, ax = plt.subplots()
df.Survived.value_counts().plot(kind='barh', color="blue", alpha=.65)
# set_ylim(-1, len(df.Survived.value_counts())) 
plt.title("Survival Breakdown (1 = Survived, 0 = Died)")



# Plotting the number of passengers per boarding count 
plt.figure(figsize=(6,4))
# fig, ax = plt.subplots()
df.Embarked.value_counts().plot(kind='bar', alpha=0.55)
# set_xlim(-1, len(df.Embarked.value_counts()))
# specifies the parameters of our graphs
plt.title("Passengers per boarding location")


# A scatterplot between the people survived and their age
plt.scatter(df.Survived, df.Age, alpha=0.55)
# sets the y axis lable
plt.ylabel("Age")
# formats the grid line style of our graphs                          
plt.grid(b=True, which='major', axis='y')  
plt.title("Survial by Age,  (1 = Survived)")
plt.show()

#  A bar plot to see who see who survived with respect to male and female count

fig  = plt.figure(figsize =(18,6))

ax1 = fig.add_subplot(121)
df.Survived[df.Sex == 'male'].value_counts().plot(kind='barh',label='Male')
df.Survived[df.Sex == 'female'].value_counts().plot(kind='barh', color='#FA2379',label='Female')
plt.title("Who Survived? with respect to Gender, (raw value counts) "); plt.legend(loc='best')


ax2 = fig.add_subplot(122)

(df.Survived[df.Sex == 'male'].value_counts()/float(df.Sex[df.Sex == 'male'].size)).plot(kind='bar',label='Male')
(df.Survived[df.Sex == 'female'].value_counts()/float(df.Sex[df.Sex == 'female'].size)).plot(kind='barh', color='#FA2379',label='Female')
plt.title("Who Survived? with respect to Gender, (proportions) "); plt.legend(loc='best')

plt.show()











