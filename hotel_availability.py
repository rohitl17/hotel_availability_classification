#!/usr/bin/env python
# coding: utf-8


# In[ ]:


# To install packages that are not installed by default, uncomment the last two lines 
# of this cell and replace <package list> with a list of necessary packages.
# This will ensure the notebook has all the dependencies and works everywhere.

#import sys
#!{sys.executable} -m pip install <package list>


# In[144]:


#Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, auc, accuracy_score, roc_curve

pd.set_option("display.max_columns", 101)


# ## Data Description

# Column | Description
# :---|:---
# `id` | The unique ID assigned to every hotel.
# `region` | The region in which the hotel is located..
# `latitude` | The latitude of the hotel.
# `longitude` | The longitude of the hotel.
# `accommodation_type` | The type of accommodation offered by the hotel. For example: Private room, Entire house/apt, etc.
# `cost` | The cost of booking the hotel for one night. (in \$\$)
# `minimum_nights` | The minimum number of nights stay required.
# `number_of_reviews` | The number of reviews accumulated by the hotel.
# `reviews_per_month` | The average number of reviews received by the hotel per month.
# `owner_id` | The unique ID assigned to every owner. An owner can own multiple hotels.
# `owned_hotels` | The number of hotels owned by the owner.
# `yearly_availability` | It indicates if the hotel accepts bookings around the year. Values are 0 (not available for 365 days in a year) and 1 (available for 365 days in a year).

# ## Data Wrangling & Visualization

# In[4]:


# Dataset is already loaded below
data = pd.read_csv("train.csv")


# In[5]:


data.head()


# In[6]:


#Explore columns
data.columns


# In[7]:


#Description
data.describe()


# In[52]:


## Printing the unique regions in the dataset
np.unique(data['region'])


# In[53]:


## Sanity check if all the hotels are unique
len(data), len(np.unique(data['id']))


# In[54]:


## Checking if the owners are unique
len(data), len(np.unique(data['owner_id']))


# ### Dataset Insights:
# -> Since the latitude and longitude values do not vary much, the data is from a specific region. The names of the regions suggest somewhere close to New York City. We'll keep the latitudes and logitudes to be on the safer side and not lose out on intricate details.
# -> The cost potentially with the amount of variation could be an important factor for predicting the outcome. At the same time, the maximum value suggest that the data could have some anomalous values, which histograms will help in figuring out
# -> The reviews per month has 676 missing values. At this point, since the business problem we are trying to solve of predicting if the hotel accepts bookings throughout the year, the average value per month is not important because it does not take into account the skewness of the monthly data, where lets say the hotel does not take bookings in July, we can't figure that out. We can think of dropping this column. Also, statistically the reviews_per_month feature does not have a lot of variation in values, hence intuitively might not contribute to the predictability of our algorithm
# -> Assuming the train and test data has no overlapsof hotels, we can safely remove the id column
# -> The mean of the yearly_availability column shows that the data is balanced at this point with a prevalence of almost 50%

# In[55]:


filtered_data=data.drop(columns=['id', 'reviews_per_month'], axis=1)
filtered_data.columns


# ## Analyze cost and minimum_nights

# #### Analyze Cost First

# In[56]:


# Plot histogram for cost values
plt.hist(filtered_data['cost'])


# In[57]:


## The data looks skewed with majority of the values in the the [0,1000] range from the histogram.
## We know the data is an independent sample of hotels, asssume the population mean of costs would follow a normal distribution,
# let's take two standard deviations to the right of the mean as it showcases in the right-tailed Z-tests.

cost_maximum_value=195.94+2*406.184
print (cost_maximum_value)

count_of_hotels_in_range=0

for hotel_cost in list(filtered_data['cost']):
    if hotel_cost<=value:
        count_of_hotels_in_range+=1

print (len(filtered_data['cost']), count_of_hotels_in_range)


# In[58]:


# We now get rid of anomalous values in the dataset by just losing out on 41 values.
#Filtering the data accordingly

filtered_data=filtered_data.loc[filtered_data['cost']<=cost_maximum_value]
print (len(filtered_data), np.min(filtered_data['cost']), np.max(filtered_data['cost']))


# ### Analyze minimum_nights now 

# In[60]:


plt.hist(filtered_data['minimum_nights'])


# In[61]:


np.unique(filtered_data['minimum_nights'])


# In[63]:


# Use same normal distribution logic to filter out samples

minimum_nights_maximum_value=11.53+2*37.97
print (minimum_nights_maximum_value)

count_of_hotels_in_range=0

for hotel_nights in list(filtered_data['minimum_nights']):
    if hotel_nights<=minimum_nights_maximum_value:
        count_of_hotels_in_range+=1

print (len(filtered_data['minimum_nights']), count_of_hotels_in_range)


# We now get rid of anomalous values for minimum nights column too, just losing out on 52 datapoints.
# Let's apply this filter on the filtered_data dataframe


# In[64]:


filtered_data=filtered_data.loc[filtered_data['minimum_nights']<=minimum_nights_maximum_value]
print (len(filtered_data), np.min(filtered_data['minimum_nights']), np.max(filtered_data['minimum_nights']))


# ## Change string to float for regions & accomodation_type

# In[162]:


def get_region_id_column(filtered_data):
    unique_regions=list(np.unique(filtered_data['region']))
    unique_region_id_mapping={}
    for idx, i in enumerate(unique_regions):
        unique_region_id_mapping[i]=idx

    print (unique_region_id_mapping)

    region_id=[]
    for i in filtered_data['region']:
        region_id.append(unique_region_id_mapping[i])
    filtered_data['region_id']=region_id

    filtered_data=filtered_data.drop(columns=['region'], axis=1)
    filtered_data.columns
    
    return filtered_data

filtered_data=get_region_id_column(filtered_data)


# In[163]:


def get_accomodation_id_column(filtered_data):
    unique_accomodations=list(np.unique(filtered_data['accommodation_type']))
    unique_accomodation_id_mapping={}
    for idx, i in enumerate(unique_accomodations):
        unique_accomodation_id_mapping[i]=idx

    print (unique_accomodation_id_mapping)

    unique_accomodation_id=[]
    for i in filtered_data['accommodation_type']:
        unique_accomodation_id.append(unique_accomodation_id_mapping[i])
    filtered_data['accomodation_id']=unique_accomodation_id

    filtered_data=filtered_data.drop(columns=['accommodation_type'], axis=1)
    filtered_data.columns
    
    return filtered_data

filtered_data=get_acccomodation_id_column(filtered_data)


# In[87]:


sns.pairplot(filtered_data, kind="scatter", hue="yearly_availability", markers=["o", "s"], palette="Set2")


# In[ ]:


# When we look at the above pairplot as a whole in majority of feature plots there we see two clusters of orange and green depicting our classes
# The data looks more separable at this point.


# ## Visualization, Modeling, Machine Learning
# 
# Build a model that categorizes hotels on the basis of their yearly availability.  Identify how different features influence the decision. Please explain the findings effectively to technical and non-technical audiences using comments and visualizations, if appropriate.
# - **Build an optimized model that effectively solves the business problem.**
# - **The model will be evaluated on the basis of Accuracy.**
# - **Read the test.csv file and prepare features for testing.**

# In[88]:


training_data=filtered_data
training_data.head()


# In[89]:


print ("Prevalence:", sum(training_data['yearly_availability'])/len(training_data)) #Looks balanced


# In[90]:


outcome=training_data['yearly_availability']
training_data=training_data.drop(['yearly_availability'], axis=1)


# ## Experimentation with hyperparameter grid search using cross-validation

# Choosing optimal hyperparamters is an important part of training the machine learning model. What we are doing here
# is with a set of specified values, the model is running on combinations of all of them and then will output the optimal set of hyperparameters
# for our dataset
# 
# Cross-validation essentially means to divide the dataset into k-sets and hold one out in each new model training iteration. The benefit of doing that is we know if the model is not succeeding by chance on our splits
# 
# Reason for choosing Random Forests:
# -> Ensemble of decision trees (Bagging - End up providing the best combination of trees)
# -> Reduced overfitting compared to decision trees
# -> Explainable as compared to paramteric models like Logistic Regression, SVMs, etc.
# -> Known to have performed quite well on tabular datasets and hence adopted for widespread use in the industry
# -> Handles missing values and continuous and categorical values quite well

# In[92]:


rfc=RandomForestClassifier(random_state=42)
param_grid = { 
    'n_estimators': [100, 1000],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [1,2,3,4,5,6,7,8,9,10],
    'criterion' :['gini', 'entropy']
}

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=4, refit='accuracy')

CV_rfc.fit(training_data, outcome)
print (CV_rfc.best_params_)


# ## Let's make stratified splits

# In[96]:


x_train, x_val, y_train, y_val = train_test_split(training_data, outcome, test_size=0.25, random_state=42)


# ## Train model

# In[97]:


rfc_model=RandomForestClassifier(random_state=42, criterion='entropy', max_depth=8, max_features='auto', n_estimators=1000, class_weight='balanced')
rfc_model.fit(x_train, y_train)


# ## Run Predictions on the Validation set

# In[141]:


predicted_probabilities = rfc_model.predict_proba(x_val)


# In[145]:


fpr, tpr, threshold = roc_curve(y_val, predicted_probabilities [:,1])
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[146]:


threshold=0.55   # Threshold chosen on the basis of the ROC curve to get maximum accuracy
predictions = (predicted_probabilities [:,1] >= threshold).astype('int')


# In[147]:


Y_test=list(y_val)
Y_pred=list(predictions)

print("Confusion Matrix:")
print (confusion_matrix(Y_test,Y_pred))

print("Classification report:")
print(classification_report(Y_test,Y_pred,target_names=['Does not accept 365 days', 'Accepts 365 days']))

print ("Acuracy:", accuracy_score(Y_test, Y_pred))


# ## Predicting on train set to check if the model is overfitting

# In[148]:


predicted_probabilities = rfc_model.predict_proba(x_train)

predictions = (predicted_probabilities [:,1] >= threshold).astype('int')


# In[149]:


Y_test=list(y_train)
Y_pred=list(predictions)

print("Confusion Matrix:")
print (confusion_matrix(Y_test,Y_pred))

print("Classification report:")
print(classification_report(Y_test,Y_pred,target_names=['Does not accept 365 days', 'Accepts 365 days']))

print ("Acuracy:", accuracy_score(Y_test, Y_pred))


# ##### The training and validation sets do not have much of a difference in their metrics. The model seems to have fitted optimally. Additional overfitting can only be checked on an out-of-sample test set
# 
# ##### Training Accuracy: 0.95
# ##### Validation Accuracy: 0.93

# In[ ]:


#Loading Test data
test_data=pd.read_csv('test.csv')
test_data.head()


# # Clean CSV

# In[160]:


filtered_test_data=test_data.drop(columns=['reviews_per_month'], axis=1)
filtered_test_data.columns


# In[164]:


filtered_test_data=get_region_id_column(filtered_test_data)
filtered_test_data=get_accomodation_id_column(filtered_test_data)


# In[165]:


filtered_test_data.head()


# In[171]:


test_predicted_probabilities = rfc_model.predict_proba(filtered_test_data.drop(['id'], axis=1))


# In[172]:


test_predictions = (predicted_probabilities [:,1] >= threshold).astype('int')


# In[175]:


positive_predictions=sum(test_predictions)/len(test_predictions)
print (positive_predictions)

## The positive prediction prevalence atleast matches the prevalence of our training dataset. 
## This does not ensure accuracy of the model, but atleast tells us the predictions are not going completely random 


# 
# 
# **Highlight the most important features of the model for management.**
# 
# > #### Task:
# - **Visualize the top 20 features and their feature importance.**
# 

# In[150]:


importances = list(rfc_model.feature_importances_)

feature_list = training_data.columns

feature_importances = [(feature, round(importance*100, 2)) for feature, importance in zip(feature_list, importances)]

feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)


# In[153]:


ft_imp_mapping = {}
for i in feature_importances:
    ft_imp_mapping[i[0]] = i[1]


# In[154]:


pd.DataFrame(ft_imp_mapping, index=["Feature Importance (%)"]).T


# In[159]:


plt.bar(ft_imp_mapping.keys(), ft_imp_mapping.values(), color='g')


# #### As we see on the basis of the importance of features, the accomodation_id corresponding to accomodation_type is the most important feature for determining if the hotel accepts the bookings throughout the year. That feature has 50% of the role to play in each of the decisions. Going on, we see the number of hotels owned by an owner, number of reviews and minimum_nights contribute to decision making. As we suspected during data cleaning the regions are not very far from each other, we see that the geospatial data has less of a role to play in decision making with latitude, longitude, and regions being less contributive to the final decision making.

# > #### Task:
# - **Submit the predictions on the test dataset using your optimized model** <br/>
#     For each record in the test set (`test.csv`), predict the value of the `yearly_availability` variable. Submit a CSV file with a header row and one row per test entry.
# 
# The file (`submissions.csv`) should have exactly 2 columns:
#    - **id**
#    - **yearly_availability**

# In[179]:


submission_df=pd.DataFrame()

submission_df['id']=filtered_test_data['id']
submission_df['yearly_availability']=test_predictions


# In[180]:


submission_df.head()


# In[181]:


#Submission
submission_df.to_csv('submissions.csv',index=False)


# ---
