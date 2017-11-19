# Importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import skew
import os 

# Changing directory
os.getcwd()
os.chdir("/Users/manishshegokar/Documents/Manish_documents/My independent projects/Python_Dataset/house prices/housing")

# Importing train and test data files
train = pd.read_csv("train.csv")

test = pd.read_csv("test.csv")

train.describe()
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))
#Data Preprocessing
train['SalePrice'].hist()

#creating data frame with old saleprice having name as prices and new salprice 
#with log name
prices = pd.DataFrame({"price":train["SalePrice"], 
"log(price+1)":np.log1p(train["SalePrice"])})
prices.hist()

train['SalePrice'] = np.log1p(train['SalePrice'])

#log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

all_data = pd.get_dummies(all_data)

all_data = all_data.fillna(all_data.mean())

#creating matrices for sklearn:
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.SalePrice

# Importing libraries
from sklearn.linear_model import Ridge, RidgeCV, LassoCV, LassoLarsCV
from sklearn import cross_validation

# Function to return rmse
def rmse_cv(model):
    rmse= np.sqrt(-cross_validation.cross_val_score(model, X_train, y, scoring="mean_squared_error", cv = 5))
    return(rmse)

model_ridge = Ridge()

# Various values for alpha
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]

cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 
            for alpha in alphas]

# fitting model on train dataset
newridge = Ridge(alpha=5).fit(X_train,y)
rmse_cv(newridge).mean()

# Predicting test dataset
p = np.expm1(newridge.predict(X_test))
result = pd.DataFrame({"ID":test.Id,"SalePrice":p})

# Saving result into result.csv
result.to_csv("result.csv",index=False)

# Plot
cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation - Just Do It")
plt.xlabel("alpha")
plt.ylabel("rmse")

cv_ridge.min()

# Lasso regression
model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)

rmse_cv(model_lasso).mean()
coef = pd.Series(model_lasso.coef_, index = X_train.columns)

print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])
                     
# residuals
matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)

# Calculating residuals of train set
preds = pd.DataFrame({"preds":model_lasso.predict(X_train), "true":y})
preds["residuals"] = preds["true"] - preds["preds"]

# Plot
preds.plot(x = "preds", y = "residuals",kind = "scatter")                     
              
# Predicting test dataset using lasso       
preds = np.expm1(model_lasso.predict(X_test))

# Saving the result
solution = pd.DataFrame({"id":test.Id, "SalePrice":preds})
solution.to_csv("sol.csv", index = False)                     

