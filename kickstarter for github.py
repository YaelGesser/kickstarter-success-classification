import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost
from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score

# Import dataset
df = pd.read_csv("/kaggle/input/kickstarter-campaigns-dataset-20/Kickstarter Campaigns DataSet.csv")
df.head()
df.describe()
nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')

df.columns
df.info()
df.describe()

We can see there are no nulls, some columns are numeric and others are of type object.

Let's check for duplicated rows and remove them. In order to do this, first we wll remove column 'Unnamed: 0 ' as it is an index that will not allow us to detect duplicates:

df=df.drop(['Unnamed: 0'], axis=1)
print("duplicates: ", df.duplicated().sum())
df.drop_duplicates(inplace=True, ignore_index=True)

# original row-length was 217245. check after removal of duplicates
print("length of dataset after removal of duplicates is ", len(df))

Some columns will not be useful in our prediction models. These will be removed:

# id column is unuseful

# We will also have no use for creator_id

# Currency data is partially represented in the country data, uniting European countries to Euro currency. Therefore we will remove
# the currency column.

# Also, the column usd_pledged can be treated just the same as the 'status' target column.
# If amount pledged is greater than the goal, obviously the status will be success. If it is less than that goal
# the status will be failed, so this column will be ommited.

# This dataset is quite large, and the blurb column is different for each row. Since we have the blurb_length column we will remove
# the blurb column.

# Since we have campaign duration in days, we will drop the end date in the column 'deadline'


columns_to_drop = ['id', 'creator_id', 'currency', 'usd_pledged', 'blurb', 'deadline' ]

df=df.drop(columns=columns_to_drop, axis=1)

Move
on
to
checking
more
columns, as the
goal is to
transform
them
all
to
numerical
data
for the prediction models.

# from 'launched_at' column we will only take the year information and ignore the month and day info

df['launched_at'] = df['launched_at'].str[:4]

# Let's see how the kikstart projects were spread over the years

year_count = df['launched_at'].value_counts()
sns.set(style="darkgrid")
sns.barplot(year_count.index, year_count.values, alpha=0.9)
plt.title('Kickstarter Distribution of Launch Year')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Year', fontsize=12)
plt.xticks(rotation=90, horizontalalignment="center")
plt.show()

# These will be now be treated as categories

# Now let's see how many countries there are and how they are spread

country_count = df['country'].value_counts()
sns.set(style="darkgrid")
sns.barplot(country_count.index, country_count.values, alpha=0.9)
plt.title('Kickstarter Distribution of Countries')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xticks(rotation=90, horizontalalignment="center")
plt.xlabel('Country', fontsize=12)
plt.show()

# We notice that the distribution of countries is highly skewed, since the vast majority of projects originates from the US. We will tend to this later.

# Let's check cities
print("The projects come from {} different cities".format(df['city'].nunique()))
df['city'].value_counts()

# Since there are 13409 different cities, and as city data is highly specific, we will drop that column and keep only the countries.
df=df.drop(columns='city')

# Now let's see how many main-categories and sub-categories there are
print("The projects are divided into {} different main categories".format(df['main_category'].nunique()))
print("They are also divided into {} different sub-categories".format(df['sub_category'].nunique()))

df['main_category'].value_counts()
df['sub_category'].value_counts()
sub_category_count = df['sub_category'].value_counts()
sns.set(style="darkgrid")
sns.barplot(sub_category_count.index, sub_category_count.values, alpha=0.9)
plt.title('Kickstarter Distribution of Projects by Sub-Category')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Sub-Category', fontsize=12)
plt.xticks(rotation=90, horizontalalignment="center")
plt.show()


# It seens that sub categories allow for a more generalized classification. Therefore
# we will keep the sub_category and drop the main_category
df = df.drop(columns = 'main_category')

# Transforming object type columns to numerical:

# The name feauture is not very indicative, we will transform this feature to its length (number of characters)

df['name_length'] = df['name'].apply(lambda x: len(x))
df=df.drop(['name'], axis=1)

# For the slug feature we will use the number of words that compose it

df['slug_length'] = df['slug'].str.count('-') + 1
df=df.drop(['slug'], axis=1)
# Check which columns need more handling, start with 'status' which will also serve as our target
df.head()

# Change the status column values to successful -> 1, failed -> 0

df.loc[df['status'] == 'successful', 'status'] = 1
df.loc[df['status'] == 'failed', 'status'] = 0
df['status'].value_counts()

# Canceled projects will be deleted

df= df[df.status!=('canceled')]

# Live projects are still ongoing, delete these as see
# (these instances can be kept and prediction can be made, which will be checked later on when new outcome is released)

df_live=df[df.status==('live')]
df= df[df.status!=('live')]

df['status'] = df['status'].astype('int')
df['status'].value_counts()

# The goal_usd is extremely different in its values from the rest of the columns. We will apply a logarithm
# transformation on it. There are quite a few benefits to using log transform: It helps to handle skewed data
# and after transformation, the distribution becomes more approximate to normal.
# It also decreases the effect of the outliers due to the normalization of magnitude differences and the
# model become more robust. The data on which log transform is applied must have only positive values,
# therefore the goal_usd fits this transformation.

df['goal_usd'].min()
df['goal_usd'].max()
print ("The minimum goal_USD is {}, the maximum goal_USD is {}".format((df['goal_usd'].min()), (df['goal_usd'].max())))

# The gap between max and min is enormous. It will also be easier to treat this column after the log transformation

df['goal_usd']=np.log(df['goal_usd'])
print ("The minimum goal_USD after log transformation is {}, the maximum goal_USD after log transformation is {}".format((df['goal_usd'].min()), (df['goal_usd'].max())))

df['goal_usd'].plot(kind='box',xlim=(-5,20), vert=False, figsize=(25,2))


plt.title("Goal USD", fontsize=18)
plt.xlabel("Values after log transform", fontsize=14)

# There is definitely one outlier, the -5 valued goal_usd, but actually all negative values represent values that
# are 1 or bellow. These might be typos in the original data (perhaps values that are 1000 were written as 1.000
# as is the syntax in some countries). However, we cannot determine the reason for the goal_usd to be of such
# a low value as 1. In addition, values over 15 also represent extremely high goal_usd's.
# Let's check how many instances satisfy either ofthese two conditions:

# outliers:
low_outs = len(df[(df['goal_usd'] <= 0)])
print ("There are {} low outliers".format(low_outs))

high_outs = len(df[(df['goal_usd'] >15)])
print ("There are {} high outliers".format(high_outs))

# both of these can be removed
df.drop(df[(df['goal_usd'] <= 0) | (df['goal_usd'] > 15)].index, axis=0, inplace=True)

# check number of rows remaining
print ("There are {} rows in the datset after outlier removal".format(len(df)))

# One more column where outliers might appear is the backers_count. Let's check the values that it contains:

print("The minimum number of backers is: {}".format(df['backers_count'].min()))
print("The maximum number of backers is: {}".format(df['backers_count'].max()))
print("The average number of backers is: {}".format(df['backers_count'].mean()))

# Projects that were successful and have 0 backers will be considered outliers and
# Check if there are any:
len(df[(df['backers_count'] <= 0) & (df['status'] > 0)])

# 3 more categorical features now need to be encoded: Launched_at, sub_category, and country. This will be done in a number of methods:
#
# Country column will be encoded using binary encoding. Binary encoding combines Hash encoding and one-hot encoding.
# The country feature will first be converted into numerical using an ordinal encoder, and then the numbers will be
# transformed to binary numbers. The binary value will be split into different columns. Binary encoding was chosen since
# it works well when there are a high number of categories and it is efficient in feature incrementation.
#
# Sub_category column will be encoded using count/frequency encoding, replacing the category by the frequency of
# observations in the dataset.
#
# Launched_at will be encoded with one-hot encoding.
#
# Afterwards double-check all features are numerical

# country column: binary encoding


encoder= ce.BinaryEncoder(cols=['country'],return_df=True)
#Fit and Transform Data
df=encoder.fit_transform(df)
df.head()

# sub_category: count/frequency encoding

sub_category_Dict = df['sub_category'].value_counts(normalize=True)
df['encoded_sub_category'] = df['sub_category'].map(sub_category_Dict)
# drop original sub-category column
df = df.drop(['sub_category'], axis=1)

# launched_at: one-hot encoding

df = pd.get_dummies(df)

df.info()

# Before preparing the data matrix and the target vector it is important to check that our data is stilil balanced
# target-wise, after removals of duplicate rows etc. We need to make sure we have at least a 70:30 ratio between
# success(1) and failure(0) or we will have to balance the data. We see that we are OK.

df['status'].value_counts(normalize=True)

df.head()

# Continue on to prepare target and data matrix:

# target
y = np.array(df.iloc[:,7])
df=df.drop(['status'], axis=1)
# data matrix
X=np.array(df.iloc[:,0:25])

print("the shape of the data X matrix is {}, the shape of the target vector is {}".format(X.shape, y.shape))

# We will now run some models on the data. It is preferable to use KFold cross-validation, but for some of the models
# this is quite heavy in computation time. Therefore the data will also be split using train-test split and some
# predictions will be made without KFold as well.

# prepare train and test data
Xtrain, Xtest, yTrain, yTest = train_test_split(X, y)

# Begin with Naive Bayes model, using KFold with k=10

k = 10
kf = KFold(n_splits=k, shuffle=True, random_state=42)
model = GaussianNB()
result = cross_val_score(model, X, y, cv=kf)

print("Avg accuracy Naive Bayes with KFold: {}".format(result.mean()))

# try Bernoulli NB as well since this is a binary classification problem, it might be better
model = BernoulliNB()
result = cross_val_score(model, X, y, cv=kf)

print("Avg accuracy Bernoulli Naive Bayes with KFold: {}".format(result.mean()))

# This model does not classify our data properly. Naive Bayesian models are particularly useful for small & medium
# sized data sets, and this dataset is quite large. That might be one reason. Also, another limitation of Naive Bayes
# is the assumption of independent predictors. When the Naive assumption does not hold true we may get poor results.
# Let's try working with KNN model:

k = 10
kf = KFold(n_splits=k, shuffle = True, random_state=42)
model = KNeighborsClassifier(5)
# result = cross_val_score(model , X, y, cv = kf)
# print("Avg accuracy Naive Bayes with KFold: {}".format(result.mean()))

# long run-time...... resulted in:
# Avg accuracy : 0.9213982539803947

# Run KNN without KFold, check 3 neighbors as well:

KNN_classifier = KNeighborsClassifier(5)
KNN_classifier.fit(Xtrain, yTrain)
preds_KNN = KNN_classifier.predict(Xtest)

print('MAE KNN:',mean_absolute_error(yTest, preds_KNN))
print('Accuracy KNN :',accuracy_score(yTest,preds_KNN))
print('Classification report KNN:',classification_report(yTest, preds_KNN))


KNN_classifier2 = KNeighborsClassifier(3)
KNN_classifier2.fit(Xtrain, yTrain)
preds_KNN2 = KNN_classifier2.predict(Xtest)


print('MAE KNN:',mean_absolute_error(yTest, preds_KNN2))
print('Accuracy KNN :',accuracy_score(yTest,preds_KNN2))
print('Classification report KNN:',classification_report(yTest, preds_KNN2))

# Accuracy is 91.1 % with 5 neighbors, 90.7 % with 3 neighbors.
# Much better than the Naive Bayes model. Let's see if we can do better, using a decision tree in two versions,
# one with max_depth 4 and a deeper one with max_depth 8:

# Decision Tree - 2 versions with KFold 10

k = 10
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# 1. max_depth = 4

model = DecisionTreeClassifier(max_depth=4, random_state=42)
result = cross_val_score(model, X, y, cv=kf)

print("Avg accuracy decision tree with KFold and max_depth 4: {}".format(result.mean()))

# 2. max_depth = 8

model = DecisionTreeClassifier(max_depth=8, random_state=43)
result = cross_val_score(model, X, y, cv=kf)

print("Avg accuracy decision tree with KFold and max_depth 8: {}".format(result.mean()))

# Well, the decision tree achieved 93% success with max_depth 4 and 94% with max_depth 8.
# We will continue on and check random forest:

model = RandomForestClassifier(random_state=42)
result = cross_val_score(model , X, y, cv = kf)
print("Avg accuracy random forest with KFold: {}".format(result.mean()))

# Accuracy of random forest is 94.4%. Last model to check is gradient boosting:

# XGBOOST
XGB_classifier = XGBClassifier()
XGB_classifier.fit(Xtrain, yTrain)
print(XGB_classifier)

# make predictions for test data
XGB_preds = XGB_classifier.predict(Xtest)
predictions = [round(value) for value in XGB_preds]

# evaluate predictions
accuracy = accuracy_score(yTest, predictions)
print("Accuracy of XGB classifier: %.2f%%" % (accuracy * 100.0))

# It seems that this is the best accuracy we are reaching at the moment.
# To end this analysis we would like to check the feature importance in a couple of the models,
# in order to get an idea which are the most influential features of the models:

# XGBOOST

# get importance
importance = XGB_classifier.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()

# Top two most influential features are the backers_count and the goal_usd, which seems to make sense.
# Nonetheless, the feature_importance method is biased and prefers features with high cardinality,
# which these two features have. Checking the random forest as well:

random_forest_classifier = RandomForestClassifier(random_state=42)
random_forest_classifier.fit(Xtrain, yTrain)
preds_rfc = random_forest_classifier.predict(Xtest)

# get importance
importance = random_forest_classifier.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()

# In the case of random forest, the top two features are the same.
# Let's do one more experiment, and try to make the predictions based on just these two features
# (backers_count, goal_usd)

# no need to change target vector, just data matrix

X2 = np.array(df[['backers_count', 'goal_usd']])
print("the new data matrix size is: {}".format(X2.shape))
Xtrain2, Xtest2, yTrain2, yTest2 = train_test_split(X2, y)

# models using 2 features only:


KNN_classifier_2features = KNeighborsClassifier(5)
KNN_classifier_2features.fit(Xtrain2, yTrain2)
preds_KNN2 = KNN_classifier_2features.predict(Xtest2)


print('MAE :',mean_absolute_error(yTest2, preds_KNN2))
print('Accuracy KNN:',accuracy_score(yTest2,preds_KNN2))


decision_tree_classifier_2features = DecisionTreeClassifier(max_depth=4, random_state=42)
decision_tree_classifier_2features.fit(Xtrain2, yTrain2)
preds_decision_tree2 = decision_tree_classifier_2features.predict(Xtest2)


print('MAE :',mean_absolute_error(yTest2, preds_decision_tree2))
print('Accuracy decision tree:',accuracy_score(yTest2,preds_decision_tree2))

# random forest:
random_forest_classifier_2features = RandomForestClassifier(random_state=42)
random_forest_classifier_2features.fit(Xtrain2, yTrain2)
preds_random_forest2 = random_forest_classifier_2features.predict(Xtest2)


print('MAE :',mean_absolute_error(yTest2, preds_random_forest2))
print('Accuracy random forest:',accuracy_score(yTest2,preds_random_forest2))

# Results are quite good. It seems that the models rely heavily on these two features,
# and the other features just add some fine-tuning which allow us to raise the accuracy by just 1.5% more.
# This is possible largely because the dataset is large, and would not work on a small dataset.