import pandas as pd
import numpy as np
sales = pd.read_csv('E:/2020-08-18 Amila/Learning/Coursera/WK2/Predict House price/home_data.csv')
#ax1 = sales.plot.scatter(x='sqft_living',   y='price', c='DarkBlue')
                      

sales2 = sales[['sqft_living','price']]


from sklearn import cross_validation
X_train,X_test,y_train,y_test = cross_validation.train_test_split(sales2[['sqft_living']],sales2[['price']],test_size=0.3)

print (y_test)

from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(X_train, y_train)

#%%
prd = reg.predict(X_test)

print(prd)

y_test = y_test.to_numpy()

#%%
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, prd)


#%%

import pandas as pd
print(pd.__version__)
