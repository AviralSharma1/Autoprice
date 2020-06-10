import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

car_df= pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data", names =['symboling','normalized_losses','make','fuel_type','aspiration','num_of_doors','body_styles','drive_wheels','engine_location','wheel_base','length','width','height','curb_weight','engine_type','num_of_cylinders','engine_size','fuel_system','bore','stroke','compression_ratio','hp','peak_rpm','city_mpg','highway_mpg','price'])
# print(car_df.head(2).transpose())
# print(car_df.dtypes)
car_df=car_df.drop('make',axis=1)
car_df=car_df.drop('fuel_type',axis=1)
car_df=car_df.drop('engine_location',axis=1)
car_df=car_df.drop('num_of_doors',axis=1)
car_df=car_df.drop('body_styles',axis=1)
car_df=car_df.drop('engine_type',axis=1)
car_df=car_df.drop('fuel_system',axis=1)
car_df=car_df.drop('aspiration',axis=1)
car_df=car_df.drop('normalized_losses',axis=1)
car_df=car_df.drop('drive_wheels',axis=1)

# Changing datatype
car_df['cylinder']= car_df['num_of_cylinders'].replace({'one':1,'two':2 , 'three':3 , 'four': 4,'five':5 , 'six':6 ,'seven':7,'eight':8,'nine':9,'ten':10 ,'eleven':11,'twelve':12})
car_df = car_df.replace('?', 0)


car_df['bore'] = car_df['bore'].astype('float64')
car_df['stroke'] = car_df['stroke'].astype('float64')
car_df['hp'] = car_df['hp'].astype('float64')
car_df['peak_rpm'] = car_df['peak_rpm'].astype('float64')
car_df['price'] = car_df['price'].astype('float64')

print(car_df.dtypes)
# print(car_df.describe().transpose())

car_df_attr = car_df.iloc[:, 1:16]
# sns.pairplot(car_df_attr, diag_kind='kde')
sns.pairplot(car_df, diag_kind='kde')


# Separating variable and outcome
X = car_df.drop('price', axis=1)
X = X.drop('num_of_cylinders', axis=1)
y = car_df[['price']]

# Splitting train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.25 , random_state=1)


# from sklearn.impute import SimpleImputer
# X2 = SimpleImputer(missing_values=np.nan ,strategy = 'median')
# X2.fit(X)

# Training Model
regression_model = LinearRegression()
regression_model.fit(X_train, y_train)

# Viewing coeffcients
for i in range(1, 14):
    print("The coefficient for {}is {}".format(X_train.columns[i], regression_model.coef_[0][i]))

intercept = regression_model.intercept_[0]
print("The intercept is", intercept)

print("Accuracy of the model is ",100*regression_model.score(X_test, y_test))

import statsmodels.formula.api as smf
# cars = pd.concat([y_train, X_train], axis=1, join='inner')
cars = pd.concat([y_train, X_train], axis=1)
cars.head()

lmcars = smf.ols(formula= 'price ~ symboling + wheel_base + length + width + height + curb_weight +  engine_size + bore + stroke + compression_ratio + hp + peak_rpm + city_mpg + highway_mpg  + cylinder', data=cars).fit()
print(lmcars.params)
print(lmcars.summary())








