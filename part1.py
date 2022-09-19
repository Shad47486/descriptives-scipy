#usually want to import all modules you want to use in the beg
#if not able to import pip install first 
# if pip install does not work use sudo pip 
import pandas #data manipulation
import numpy as np #numerical computing, optimizing arrays 
from scipy import stats #simple statistical tests
import seaborn #visualization tool for data
from pandas import plotting
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols

#reading data from folder:
data = pandas.read_csv('examples/brain_size.csv', sep=';', na_values=".")
data

#creates an array:
t = np.linspace(-6, 6, 20)
sin_t = np.sin(t)
cos_t = np.cos(t)
#data frame using pandas 
pandas.DataFrame({'t': t, 'sin': sin_t, 'cos': cos_t})

# This is data manipulation:
data.shape    # 40 rows and 8 columns
data.columns
print(data['Gender'])
data[data['Gender'] == 'Female']['VIQ'].mean()

#group by= splitting a data frame on values of categorical variables
groupby_gender = data.groupby('Gender')
for gender, value in groupby_gender['VIQ']:
    print((gender, value.mean()))
groupby_gender.mean()

#For a box plot example3.1.6.2:
#Scatter matrices for different columns
# uses pandas plotting tool so: from pandas import plotting
plotting.scatter_matrix(data[['Weight', 'Height', 'MRI_Count']])
plotting.scatter_matrix(data[['PIQ', 'VIQ', 'FSIQ']])
plt.show()

#from scipy import stats (pre):
##tests if the mean of data is likely to be equal to a given value 
stats.ttest_1samp(data['VIQ'], 0)
# testing for differences across population between males and females
female_viq = data[data['Gender'] == 'Female']['VIQ']
male_viq = data[data['Gender'] == 'Male']['VIQ']
stats.ttest_ind(female_viq, male_viq)

#Paried Test: Repeated Measuremtns on the same Individuals:
stats.ttest_ind(data['FSIQ'], data['PIQ']) #can be a problem since it can forget links between observations 
stats.ttest_rel(data['FSIQ'], data['PIQ']) #variance is confounding, and can be removed using "paired test", or "repeated measures test"
stats.ttest_1samp(data['FSIQ'] - data['PIQ'], 0) #equiv to a 1 sample test
stats.wilcoxon(data['FSIQ'], data['PIQ']) #wilcoxon signed rank test
# Wilcoxon test:
# test the location of a population based on a sample of data, or to compare the locations of two populations using two matched samples.


#Simple Linear Regression: 
# 2 sets of obs, x and y, and we 
# want to test the hypothesis that y is a linear function of x aka: 
# y = x * (coef) + (intercept) + e 
#already imported numpy as np 
    # 1st you need to generate simulated data according to the model: 
x = np.linspace(-5, 5, 20) 
np.random.seed(1)
    #Normal Distributed noise 
y = -5 + 3*x + 4 * np.random.normal(size=x.shape)
    # this creates a data frame that contains all the relevent variables
data = pandas.DataFrame({'x': x, 'y': y})
model = ols("y ~ x", data).fit()
print(model.summary()) # Shows regression results from OLS

#categorical variables: comparing groups or multiple categories:
data = pandas.read_csv('examples/brain_size.csv', sep=';', na_values=".")
model = ols("VIQ ~ Gender + 1", data).fit() # comparision between IQ of male vs female WITH A LINEAR MODEL
#'gender' is atomatically detected as categorical aka meaning is values will be treated as different entities
print(model.summary())
model = ols('VIQ ~ C(Gender)', data).fit()
# "Gender" is Categorical
print(model.summary()) 

#Link to t-test between different FSIQ and PIQ:
data_fisq = pandas.DataFrame({'iq': data['FSIQ'], 'type': 'fsiq'})
data_piq = pandas.DataFrame({'iq': data['PIQ'], 'type': 'piq'})
data_long = pandas.concat((data_fisq, data_piq))
print(data_long)  
model = ols("iq ~ type", data_long).fit()
print(model.summary())
stats.ttest_ind(data['FSIQ'], data['PIQ']) 

#Multiple Regression (Multiple Factors):
# z = x(c1)+y(c2)+i+e
# linear model explaining a variable z (dependent var) with 2 variables x and y aka making a 3d plane of a linear model
data = pandas.read_csv('examples/iris.csv')
model = ols('sepal_width ~ name + petal_length', data).fit()
print(model.summary()) 

#Analysis of variance(ANOVA):
# Testing a diffference between the coefficent assoicated to verscolor and virginica in the linear model in the iris.csv
print(model.f_test([0, 1, -1, 0]))  

#Pairplot: scatter matrices:
# using seaborn we can easily see the interaction between continous variables
import seaborn
seaborn.pairplot(data, vars=['WAGE', 'AGE', 'EDUCATION'],
                 kind='reg')
#Categorical varibles can be plotted as the hue 
seaborn.pairplot(data, vars=['WAGE', 'AGE', 'EDUCATION'],
                 kind='reg', hue='SEX')
from matplotlib import pyplot as plt
plt.rcdefaults()


#implot:plotting a uinivariate regression
# regression caputing the relation between one var and another using the command:
seaborn.lmplot(y='WAGE', x='EDUCATION', data=data)


#testing for interaction:
result = sm.ols(formula='wage ~ education + gender + education * gender',
                data=data).fit()    
print(result.summary())