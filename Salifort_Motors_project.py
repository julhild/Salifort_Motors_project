
# ## **Pace: Plan**
# 
# Consider the questions in your PACE Strategy Document to reflect on the Plan stage.
# 
# In this stage, consider the following:

# ### Understand the business scenario and problem
# 
# The HR department at Salifort Motors wants to take some initiatives to improve employee satisfaction levels at the company. They collected data from employees, but now they don’t know what to do with it. They refer to you as a data analytics professional and ask you to provide data-driven suggestions based on your understanding of the data. They have the following squestion: what’s likely to make the employee leave the company?
# 
# Your goals in this project are to analyze the data collected by the HR department and to build a model that predicts whether or not an employee will leave the company.
# 
# If you can predict employees likely to quit, it might be possible to identify factors that contribute to their leaving. Because it is time-consuming and expensive to find, interview, and hire new employees, increasing employee retention will be beneficial to the company.

# ### Familiarize yourself with the HR dataset
# 
# The dataset that you'll be using in this lab contains 15,000 rows and 10 columns for the variables listed below. 
# 
# **Note:** you don't need to download any data to complete this lab. For more information about the data, refer to its source on [Kaggle](https://www.kaggle.com/datasets/mfaisalqureshi/hr-analytics-and-job-prediction?select=HR_comma_sep.csv).
# 
# Variable  |Description |
# -----|-----|
# satisfaction_level|Employee-reported job satisfaction level [0&ndash;1]|
# last_evaluation|Score of employee's last performance review [0&ndash;1]|
# number_project|Number of projects employee contributes to|
# average_monthly_hours|Average number of hours employee worked per month|
# time_spend_company|How long the employee has been with the company (years)
# Work_accident|Whether or not the employee experienced an accident while at work
# left|Whether or not the employee left the company
# promotion_last_5years|Whether or not the employee was promoted in the last 5 years
# Department|The employee's department
# salary|The employee's salary (U.S. dollars)

# 💭
# ### Reflect on these questions as you complete the plan stage.
# 
# *  Who are your stakeholders for this project?
#      - The stakeholders for this project are the HR department of the Saliford Motors company.    
# - What are you trying to solve or accomplish?
#      - We are trying to get insights into the employee turn-over and factors leading to it. We need to find out the reasons behind such a larger rate employees leaving and possibly find ways to retain employees longer at the company.
# - What are your initial observations when you explore the data?
#      - There are 10 recorded variables, 8 are numerical and 2 are categorical. All of them seem to be important in reagard to the employeee turn-over. It is also a self-reported data directly from emplyees, therefore it is beneficial to look at it coming directly from the source.
# - What resources do you find yourself using as you complete this stage? (Make sure to include the links.)
#     - The discription of the project and data on the coursera.
# - Do you have any ethical considerations in this stage?
#     - There are no ethical concerns regarding employee related data: no data about gender, age, no names or employee ids, therefore there will be no bias based on those values.
#     - There are some concerns about business impact from the analysis findings. If they are incorrect it can result in incorrected and somewhere inappropriate understanding of the employee dynamics and needs. 

# ## Step 1. Imports
# 
# *   Import packages
# *   Load dataset
# 
# 

# ### Import packages

# In[1]:


# Import packages

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# For displaying all of the columns in dataframes
pd.set_option('display.max_columns', None)

import pickle

# xg boost modeling
from xgboost import XGBClassifier, XGBRegressor, plot_importance

# logistic regression method
from sklearn.linear_model import LogisticRegression

# decision tree methods
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

# building a model
from sklearn.model_selection import GridSearchCV, train_test_split

# model metrics
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score,    ConfusionMatrixDisplay, confusion_matrix, classification_report, roc_auc_score, roc_curve 


# ### Load dataset

# Load dataset into a dataframe
df0 = pd.read_csv("HR_capstone_dataset.csv")

# Display first few rows of the dataframe
df0.head()


# ## Step 2. Data Exploration (Initial EDA and data cleaning)
# 
# - Understanding variables
# - Clean your dataset (missing data, redundant data, outliers)


# ### Gather basic information about the data

# In[3]:


# Gather basic information about the data
df0.info()


# ### Gather descriptive statistics about the data

# In[4]:


# Gather descriptive statistics about the data
df0.describe()


# ### Rename columns

# As a data cleaning step, rename the columns as needed. Standardize the column names so that they are all in `snake_case`, correct any column names that are misspelled, and make column names more concise as needed.

# In[5]:


# Display all column names
df0.columns


# In[6]:


# Rename columns as needed
df0 = df0.rename(columns = {
    'Work_accident': 'work_accident',
    'Department': 'department',
    'average_montly_hours': 'average_monthly_hours',
    'time_spend_company': 'tenure'
})


# Display all column names after the update
df0.columns


# ### Check missing values

# Check for any missing values in the data.

# In[7]:


# Check for missing values
df0.isna().sum()

# ### Check duplicates

# Check for any duplicate entries in the data.

# In[8]:


# Check for duplicates
df0.duplicated().sum()


# In[9]:


# Inspect some rows containing duplicates as needed
df0[df0.duplicated()].head()


# In[10]:


# Drop duplicates and save resulting dataframe in a new variable as needed
df1 = df0.drop_duplicates(keep='first')

# Display first few rows of new dataframe as needed
df1.head()


# ### Check outliers

# Check for outliers in the data.

# In[11]:


# Create a boxplot to visualize distribution of `tenure` and detect any outliers
plt.figure(figsize=(8,5))
plt.title('Box plot to check for tenure outliers', fontsize=12);
sns.boxplot(x = df1['tenure'])
plt.show()


# In[12]:


# Determine the number of rows containing outliers
tenure25 = df1['tenure'].quantile(0.25)
tenure75 = df1['tenure'].quantile(0.75)

tenure_range = tenure75 - tenure25

lower_value = tenure25 - 1.5 * tenure_range
upper_value = tenure75 + 1.5 * tenure_range

print('Lower limit for tenure:', lower_value)
print('Upper limit for tenure:', upper_value)

tenure_outliers = df1[(df1['tenure'] < lower_value) | (df1['tenure'] > upper_value)]
percent_value = len(tenure_outliers) / len(df1) * 100;

print('Number of outliers for tenure:', len(tenure_outliers), "or {percent:.2}%".format(percent = percent_value))


# Certain types of models are more sensitive to outliers than others. When you get to the stage of building your model, consider whether to remove outliers, based on the type of model you decide to use.

# # PACE: Analyze Stage
# - Perform EDA (analyze relationships between variables)
# 
# 

# 💭
# ### Reflect on these questions as you complete the analyze stage.
# 
# - What did you observe about the relationships between variables?
# - What do you observe about the distributions in the data?
# - What transformations did you make with your data? Why did you chose to make those decisions?
# - What are some purposes of EDA before constructing a predictive model?
# - What resources do you find yourself using as you complete this stage? (Make sure to include the links.)
# - Do you have any ethical considerations in this stage?
# 
# 
# 

# ## Step 2. Data Exploration (Continue EDA)
# 
# Begin by understanding how many employees left and what percentage of all employees this figure represents.

# In[13]:


# Get numbers of people who left vs. stayed
print(df1['left'].value_counts())

# Get percentages of people who left vs. stayed
print(df1['left'].value_counts(normalize=True))


# ### Data visualizations

# Now, examine variables that you're interested in, and create plots to visualize relationships between variables in the data.

# In[14]:


# Average monthly hours vs number of projects

fig, ax = plt.subplots(1, 2, figsize=(22,8));

# box plot: shows the spread and outliers
sns.boxplot(data=df1, x='average_monthly_hours', y='number_project', hue='left', orient='h', ax=ax[0]);
ax[0].invert_yaxis();
ax[0].set_title('Box plot for monthly hours by number of projects', fontsize='16');
ax[0].set_xlabel('Monthly hours on average', fontsize='14');
ax[0].set_ylabel('Number of projects', fontsize='14');

# histogram: shows numbers in comparison
sns.histplot(data=df1, x='number_project', hue='left', multiple='dodge', shrink=2, ax=ax[1]);
ax[1].set_title('Employee turnover by number of projects', fontsize='16');
ax[1].set_xlabel('Number of projects', fontsize='14');
ax[1].set_ylabel('Number of employees', fontsize='14');
ax[1].legend(labels=['Left', 'Stayed']);


# - As the box plot shows, the more projects an employee has, the more hours on average they will work. This was to expect.
# - As both plots shows there are two groups of employees leaving the company: 
#     - Group I are employees who have only two projects and worked the least hours pro months.
#     - Group II are emplyees who worked on 6 or 7 projects and were working more than 250 hours a month (burn out?)
# - Taking 20 working days per month and 8 hours per working day gives us 160 hours per month. In all categories the average number of hours worked per month is much higher starting at 185 for employees with 2 projects. Group I of the emplyees did on average 145 hours per months. Maybe it is a group of emloyees that where let go of not making the hours. Otherwise all other employees are clearly overworked.
# - People working on 3 or 4 projects are the ones most unlikely to leave.
# - All employees who worked on 7 project left, s. confirmation below.

# In[15]:


empl_7projects = len(df1[df1['number_project'] == 7])
print("Number of employees with 7 projects:", empl_7projects)

empl_7projects_left = len(df1[(df1['number_project'] == 7) & (df1['left']==1)])
print("Number of employees with 7 projects, who left:", empl_7projects_left)


# In[16]:


# Scatter plot for staisfaction level by monthly hours

plt.figure(figsize=(12,8));
sns.scatterplot(data=df1, x='average_monthly_hours', y='satisfaction_level', hue='left', alpha=0.5);
plt.title("Satisfaction level by monthly hours");
plt.ylabel("Satisfaction level");
plt.xlabel("Average monthly hours");
plt.axvline(x=160, color= 'green', label = '160 hours per month', ls = '-')
plt.legend(labels=['160 hr/month', 'stayed', 'left']);

plt.show();


# Scatter plot for satisfaction level as a function of average monthly hours revealed that there are three groups of employees who left:
# - Overworked employees with 240 up to 320 working hours per months and low satisfactory levels (around 0.1).
# - Hard working emplyees with 215 to 275 working hours per months and high satisfaction levels (0.7 - 0.9). It could be that those employees where otherwise satisfied but still chose to look for a better position (not scared to leave the comfort zone). Possibly the ground for leaving were not the hours. 
# - The third group of employees was somewhat satisfied (levels of 0.3-0.5) and was working less than normal hours. It could be that it is a group of people who cannot work longer hours, but clearly expected and under pressure to do so. People with children, e.g., could be offered a part-time position or hybrid/remote working style.
#     
# As we already know, the data is generated for purely research purpose, therefore it looks unnatural (boxy).

# In[17]:


# Calculate mean and median satisfaction scores of employees who left and those who stayed
df1.groupby(['left'])['satisfaction_level'].agg([np.mean,np.median])


# In[18]:


# Satisfaction level vs tenure

fig, ax = plt.subplots(1, 2, figsize=(22,8));

# box plot: shows the spread and outliers
sns.boxplot(data=df1, x='satisfaction_level', y='tenure', hue='left', orient='h', ax=ax[0]);
ax[0].invert_yaxis();
ax[0].set_title('Satisfaction by tenure', fontsize='16');
ax[0].set_xlabel('Satisfaction level', fontsize='14');
ax[0].set_ylabel('Tenure', fontsize='14');

# histogram: shows numbers in comparison
sns.histplot(data=df1, x='tenure', hue='left', multiple='dodge', shrink=5, ax=ax[1]);
ax[1].set_title('Employee turnover by tenure', fontsize='16');
ax[1].set_xlabel('Tenure in years', fontsize='14');
ax[1].set_ylabel('Number of employees', fontsize='14');
ax[1].legend(labels=['Left', 'Stayed']);


# - The long-tenure employees have high satisfaction levels and do not leave.
# - Emplyees who left after 3-4 years have lower satisfaction levels. Maybe it may be related to them having more responsibilities and work load and not higher salary or position.
# - Employees with 5-6 years working at the company also leave, but have very high satisfaction levels. Spreading wings and confident in a tackling new challenges?

# In[19]:


# Salary vs tenure

plt.figure(figsize=(12,8));
sns.histplot(data=df1, x='tenure', hue='salary', discrete=1, shrink=0.5, multiple='dodge',);
plt.title("Satisfaction level by monthly hours");
plt.ylabel("Number of employees");
plt.xlabel("Tenure");
plt.show();


# The plot above shows that the long-term emplyees are not only the ones in the management (only high-earning). That means that employees are leaving not due to no long-term position available.

# In[20]:


# Evaluation vs monthly hours

plt.figure(figsize=(12,8));
sns.scatterplot(data=df1, x='average_monthly_hours', y='last_evaluation', hue='left', alpha=0.5);
plt.title("Last evaluation vs monthly hours");
plt.ylabel("Last evaluation");
plt.xlabel("Average monthly hours");
plt.axvline(x=160, color= 'green', label = '160 hours per month', ls = '-')
plt.legend(labels=['160 hr/month', 'left', 'stayed']);

plt.show();


# - The above plot shows again that most employees work more than the nominal 160 hours per month.
# - There are two groups of employees who leave the company:
#     - Overworked employees with a high evaluation score
#     - Emplyees working less than 160 hours per months and having medium evaluation score (0.45-0.55)
# - The most employees with high evaluation score work many hours (is it a requirement for the high evauation?).

# In[21]:


# Promotion in last 5 years vs monthly hours

plt.figure(figsize=(12,5));
sns.scatterplot(data=df1, x='average_monthly_hours', y='promotion_last_5years', hue='left', alpha=0.5);
plt.title("Promotion in last 5 years vs monthly hours");
plt.ylabel("Promoted");
plt.xlabel("Average monthly hours");
plt.axvline(x=160, color= 'green', label = '160 hours per month', ls = '-')
plt.legend(labels=['160 hr/month', 'left', 'stayed']);

plt.show();


# - The above plot shows that most of the employees who left the company worked the longest hours and were not promoted.
# - There are only few persons who were promoted and quit the company.
# - Only few employees working longest hours were promoted.

# In[22]:


# Number of employees in each department
df1['department'].value_counts()


# In[23]:


# Employees leaving pro department

plt.figure(figsize=(12,8));
sns.histplot(data=df1, x='department', hue='left', hue_order = [0, 1], discrete=1, shrink=0.5, multiple='dodge');
plt.title("Comparison for employees leaving/staying pro department");
plt.ylabel("Number of employees");
plt.xlabel("Department");
plt.xticks(rotation='45')
plt.legend(labels=['left', 'stayed']);
plt.show();


# This plot shows that there is no noticable correlation between employees leaving and the department they are working in. It seems to be a general problem and is not localized.

# In[24]:


# Correlations between the variables

plt.figure(figsize=(12,8));
corr_plot = sns.heatmap(df1.corr(), vmin=-1, vmax=1, annot =True);
corr_plot.set_title('Correlations between the variables');


# - The correlations heat map shows that 'left' variable is strongest correlated with the satisfaction level. It is a negative correlation.
# - Another important correlation is between last evaluation, average working hours and number of projects. It is a positive correlations, meaning the higher are the hours and the number of projects, the higher will be the evaluation.

# ### Insights

# - Employees leave most likely to low satisfaction levels, long working hours, and too many projects.
# - Many employees in general work over the standard level working hours.
# - Many employees working the longest hours and who left were not promoted. 
# - Employees working more than six years at the company tend to stay.

# # PACE: Construct Stage
# - Determine which models are most appropriate
# - Construct the model
# - Confirm model assumptions
# - Evaluate model results to determine how well your model fits the data
# 

# 🔎
# ## Recall model assumptions
# 
# **Logistic Regression model assumptions**
# - Outcome variable is categorical
# - Observations are independent of each other
# - No severe multicollinearity among X variables
# - No extreme outliers
# - Linear relationship between each X variable and the logit of the outcome variable
# - Sufficiently large sample size
# 
# 
# 
# 

# 💭
# ### Reflect on these questions as you complete the constructing stage.
# 
# - Do you notice anything odd?
# - Which independent variables did you choose for the model and why?
# - Are each of the assumptions met?
# - How well does your model fit the data?
# - Can you improve it? Is there anything you would change about the model?
# - What resources do you find yourself using as you complete this stage? (Make sure to include the links.)
# - Do you have any ethical considerations in this stage?
# 
# 

# ## Step 3. Model Building, Step 4. Results and Evaluation
# - Fit a model that predicts the outcome variable using two or more independent variables
# - Check model assumptions
# - Evaluate the model

# ### Identify the type of prediction task.

# The goal is to build model that can predict if the employee is going to leave the company or not using the existing data set. The data set contains a variable 'left' that is categorical and takes values of 0 (stayed) and 1 (left). This type of problem represents a **binary classification**.

# ### Identify the types of models most appropriate for this task.

# To predict a binary variable two approaches are most suitable: **logistic regression model** and a **tree-based machine learning method**.

# ## Logistic Regression model

# To build a logistic regression model we need to encode categorical variables as numerical values.
# 
# - Department is a categorical variable and needs to be encoded with dummies
# - Salary is also vcategorical, however is ordinal and can be encoded with numbers 0, 1, 2

# In[25]:


# data set with encoded variables
df_enc = df1.copy();

# encode salary
df_enc['salary'] = (df_enc['salary'].astype('category').cat.set_categories(['low', 'medium', 'high']).cat.codes)

# get dummies for department
df_enc = pd.get_dummies(df_enc, drop_first = False)

df_enc.head()


# In[26]:


# logistic regression is sensitive to outliers, therefore we need to remove them
# the only variable with outliers was tenure

df_log_reg = df_enc[(df_enc['tenure'] >= lower_value) & (df_enc['tenure'] <= upper_value)] 
df_log_reg.head()


# In[27]:


# outcome variable 
y = df_log_reg['left'];
y.head()


# In[28]:


# feature variables for logostic regression
X = df_log_reg.drop('left', axis=1)
X.head()


# In[29]:


X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.25, 
                                                        stratify=y, 
                                                        random_state=42)


# In[30]:


# build the logistic regression model
logistic_model = LogisticRegression(random_state = 42, max_iter=500).fit(X_train, y_train)


# In[31]:


y_logistic_pred = logistic_model.predict(X_test)


# In[32]:


# calculate confusion matrix for the logistic regression model

logistic_confusion_matrix = confusion_matrix(y_test, y_logistic_pred, labels = logistic_model.classes_)
display = ConfusionMatrixDisplay(confusion_matrix=logistic_confusion_matrix, 
                                  display_labels=logistic_model.classes_)

# plot confusion matrix
display.plot(values_format='')

# display plot
plt.show()


# The confusion matrix shows four values: diagonal for 
# - true negatives (correctly predicted number of employees who **will not** leave)
# - true positives (correctly predicted number of employees who **will** leave) 
# 
# and off-diagonal value for 
# - false positives (wrongly predicted number of employees that would leave)
# - false negatives (wrongly predicted number of employees that would not leave)
# The number of true negatives is much higher than the off-diagonal values, therefore model will predict if an employees stays more are correctly.
# On the other hand, the number of true positives is lower than both off-diagonals. That means that the model is not that good at predicting if an employee would leave. Which is the main goal of the model.
# 
# 
# The result can also occur if the class is unbalanced (there are not enough data on one of the left type), which as the analysis stage showed is not the case: there are 83% of entries for 'stayed' and 17% for 'left', which not a strongly unbalanced class.

# In[33]:


# classification report for logistic model
target_names = ['Would not leave', 'Would leave'];
print(classification_report(y_test, y_logistic_pred, target_names = target_names))


# Metrics for logistic regression model: 
# - Precision: 79%
# - Recall: 82%
# - F1-Score: 80%
# 
# The metrics for the logistic regression model are quite solid, however these are for both: predicting if an employee would leave/not leave. The classification report above, however, shows that the prediction strenth of the model to predict if an employee would leave, which is the main goal of this project, is not sufficient.

# ## Tree-based ML models

# Here we are going to implement two methods: **Decision Tree** and **Random Forest** Approaches

# In[34]:


y = df_enc['left'];
y.head()


# In[35]:


X = df_enc.drop('left', axis=1);
X.head()


# In[36]:


def split_data(test_size: int, seed: int):
    return train_test_split(X, y, test_size=test_size, stratify=y,random_state=seed)


# In[37]:


X_train, X_test, y_train, y_test = split_data(0.25, 0)


# In[38]:


# model metrics to calculate
scoring = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}


# In[39]:


def tree_params(max_depth, min_sample_leaf, min_sample_split):
    return {'max_depth': max_depth,
             'min_samples_leaf': min_sample_leaf,
             'min_samples_split': min_sample_split
             }


# ### Decision tree

# In[40]:


# define the tree model
tree_model = DecisionTreeClassifier(random_state=0)

# max depth, min sample leaf, min sample split
cv_params = tree_params([4,6,8, None], [1,2,4], [2,4,6])

tree_grid_result = GridSearchCV(tree_model, cv_params, scoring=scoring, cv = 4, refit='roc_auc')


# In[41]:


get_ipython().run_cell_magic('time', '', 'tree_grid_result.fit(X_train, y_train)')


# In[42]:


# best parameters from the grid search
tree_grid_result.best_params_


# In[43]:


# best score for auc
tree_grid_result.best_score_


# The best score for AUC is 96.99%, which means that the tree model can predict if an employee will leave very well.

# In[44]:


def make_result_table(model_name: str, auc: float, precision: float, accuracy: float, recall: float, f1: float):
    result_table = pd.DataFrame({
            'Model': [model_name],
            'AUC': [auc],
            'Precision': [precision],
            'Accuracy': [accuracy],
            'Recall': [recall],
            'F1': [f1]
        });
    
    return result_table;


# In[45]:


def make_results(model_name: str, model_object, metric:str):
    
    # list of metrics of interest
    metrics = {
        'auc': 'mean_test_roc_auc',
        'precision': 'mean_test_precision',
        'accuracy': 'mean_test_accuracy',
        'recall': 'mean_test_recall',
        'f1': 'mean_test_f1'
    };
    
    # model results
    results = pd.DataFrame(model_object.cv_results_);
    
    # extract the row with the best metrics values
    best_values = results.iloc[results[metrics[metric]].idxmax(), :];
    
    # best metrics
    auc = best_values.mean_test_roc_auc;
    precision = best_values.mean_test_precision;
    accuracy = best_values.mean_test_accuracy;
    recall = best_values.mean_test_recall;
    f1 = best_values.mean_test_f1;
    
    result_table = make_result_table(model_name, auc, precision, accuracy, recall, f1);
    
    return result_table;
    


# In[46]:


all_results = make_results('Decision Tree', tree_grid_result, 'auc')
all_results


# The metrics for tree model show that it has a strong predictive power. However, tree models are known to be vulnerable to overfitting, therefore, it would be better to construct a random forest model that is not prone to the overfitting.

# ### Random Forest 

# In[47]:


def random_forest_params(max_depth, max_features, max_samples, min_samples_leaf, min_samples_split, n_estimators): 
    return {
        'max_depth': max_depth,
        'max_features': max_features,
        'max_samples': max_samples,
        'min_samples_leaf': min_samples_leaf,
        'min_samples_split': min_samples_split,
        'n_estimators': n_estimators
    }


# In[48]:


random_forest = RandomForestClassifier(random_state=0)

params = random_forest_params([3, 5, None], [1.0], [0.7, 1.0], [1, 2, 3], [2, 3, 4], [300, 500]);

random_forest_model = GridSearchCV(random_forest, params, scoring = scoring, cv = 4, refit = 'roc_auc');


# In[49]:


get_ipython().run_cell_magic('time', '', 'random_forest_model.fit(X_test, y_test)')


# In[50]:


path = '/home/jovyan/work/'


# In[51]:


# function to use pickle to save model results
def write_pickle(path, model, model_name):
    with open(path + model_name + '.pickle', 'wb') as to_write:
        pickle.dump(model, to_write)


# In[52]:


# function to use pickle to save model results
def read_pickle(path, model_name):
    with open(path + model_name + '.pickle', 'rb') as to_read:
        model  = pickle.load(to_read)
        
    return model


# In[53]:


write_pickle(path, random_forest_model, 'Random_Forest');


# In[54]:


random_forest_model = read_pickle(path, 'Random_Forest');


# In[55]:


random_forest_model.best_score_


# In[56]:


random_forest_model.best_params_


# In[57]:


random_forest_results = make_results('Random Forest', random_forest_model, 'auc')
all_results = all_results.append(random_forest_results)
print(all_results)


# In[58]:


# function for scotres from the test data set
def get_test_scores(model_name: str, model, X_test_data, y_test_data):
    prediction = model.best_estimator_.predict(X_test_data)
    
    auc = roc_auc_score(y_test_data, prediction)
    precision = precision_score(y_test_data, prediction)
    accuracy = accuracy_score(y_test_data, prediction)    
    recall = recall_score(y_test_data, prediction)    
    f1 = f1_score(y_test_data, prediction)                        
                        
    table = make_result_table(model_name, auc, precision, accuracy, recall, f1);
    return table;


# In[59]:


rf_test_scores = get_test_scores('Random Forest Test', random_forest_model, X_test, y_test);
all_results = all_results.append(rf_test_scores)
print(all_results)


# The test scores for the random forest model have similar values as the validation scores, this means it is a strong stable model.

# ### Feature engineering
# To make sure there is no data leaking, we are going to remove the *satisfaction* variable from the data and introduce a new variable *overworked*.

# In[60]:


df_eng = df_enc.drop('satisfaction_level', axis = 1);
df_eng.head()


# In[61]:


# engineered variable
# start from average hours
df_eng['overworked'] = df_eng['average_monthly_hours'];
df_eng.head()


# The normal working hours would be 20 days per months multiplied by 8 hours a day, which results in 160 hours per months. Let's assume overworking means working more than 8.5 hours per day, which results in 170 hours per month.

# In[62]:


df_eng['overworked'] = (df_eng['overworked'] > 170).astype(int)
df_eng['overworked'].head()


# In[63]:


df_eng = df_eng.drop('average_monthly_hours', axis = 1);
df_eng.head()


# In[118]:


df_eng['overworked'].value_counts(normalize=True)


# In[ ]:





# In[64]:


y = df_eng['left'];
X = df_eng.drop('left', axis = 1);


# In[65]:


X_train, X_test, y_train, y_test = split_data(0.25, 0)


# ### Decision Tree Engineered

# In[66]:


# define the tree model
tree_model = DecisionTreeClassifier(random_state=0)

# max depth, min sample leaf, min sample split
cv_params = tree_params([4,6,8, None], [2,5,1], [2,4,6])

eng_tree_result = GridSearchCV(tree_model, cv_params, scoring=scoring, cv = 4, refit='roc_auc')


# In[67]:


get_ipython().run_cell_magic('time', '', 'eng_tree_result.fit(X_train, y_train)')


# In[68]:


eng_tree_result.best_params_


# In[69]:


eng_tree_result.best_score_


# The decision tree model with the engineered feature performs also very well.

# In[70]:


eng_tree_results = make_results('Eng. Decision Tree', eng_tree_result, 'auc')


# In[71]:


all_results = all_results.append(eng_tree_results)
print(all_results)


# The decision tree model has scores a bit lower than the original one, which can be explained by less number of features taken into the account. However, the score values ares still very high.

# ### Random Forest Engineered

# In[73]:


random_forest = RandomForestClassifier(random_state=0)

params = random_forest_params([3, 5, None], [1.0], [0.7, 1.0], [1, 2, 3], [2, 3, 4], [300, 500]);

random_forest_eng = GridSearchCV(random_forest, params, scoring = scoring, cv = 4, refit = 'roc_auc');


# In[74]:


get_ipython().run_cell_magic('time', '', '\nrandom_forest_eng.fit(X_train, y_train)')


# In[75]:


write_pickle(path, random_forest_eng, 'Random_Forest_Eng');


# In[ ]:


random_forest_eng = read_pickle(path, 'Random_Forest_Eng')


# In[77]:


random_forest_eng.best_score_


# In[78]:


random_forest_eng.best_params_


# In[80]:


random_forest_eng_result = make_results('Random Forest Eng', random_forest_eng, 'auc');


# In[81]:


all_results = all_results.append(random_forest_eng_result)
print(all_results)


# The random forest model trained on the engineered data set has slightly worse scores than the original, a bit better than the corresponding decision tree model though.

# In[83]:


random_forest_eng_test = get_test_scores('Random Forest Eng Test', random_forest_eng, X_test, y_test);


# In[84]:


all_results = all_results.append(random_forest_eng_test)
print(all_results)


# The test scores from the engineered random forest are similar to the validation scores, which shows that the model is stable and consistent.

# In[86]:


# calculate confusion matrix for the engineered random forest model
prediction = random_forest_eng.best_estimator_.predict(X_test)
rf_eng_matrix = confusion_matrix(y_test, prediction, labels = logistic_model.classes_)
display = ConfusionMatrixDisplay(confusion_matrix=rf_eng_matrix, 
                                  display_labels=random_forest_eng.classes_)

# plot confusion matrix
display.plot(values_format='')

# display plot
plt.show()


# The confusion matrix plot indicates that the engineered random forest model has strong predictive power if an employee will leave or will stay.

# In[89]:


# decision tree plot

plt.figure(figsize=(85,20))
plot_tree(eng_tree_result.best_estimator_, max_depth=6, fontsize=14, feature_names=X.columns, 
          class_names={0:'stayed', 1:'left'}, filled=True);
plt.show()


# In[94]:


# feature importance from engineered decision tree

eng_tree_features = pd.DataFrame(eng_tree_result.best_estimator_.feature_importances_,
                                columns=['gini_importance'],
                                 index = X.columns
                                )
eng_tree_features = eng_tree_features.sort_values(by='gini_importance', ascending = False)
eng_tree_features = eng_tree_features[eng_tree_features['gini_importance'] != 0]
eng_tree_features


# In[95]:


sns.barplot(data=eng_tree_features, x="gini_importance", y=eng_tree_features.index, orient='h')
plt.title("Decision Tree: Feature Importances for Employee Leaving", fontsize=12)
plt.ylabel("Feature")
plt.xlabel("Importance")
plt.show()


# The plot above shows that the most important features that contribute to the employees leaving are: last evaluation, number of projects, tenure and overworked.

# In[97]:


# feature importance from engineered random forest
eng_rf_features = random_forest_eng.best_estimator_.feature_importances_;
top10_indices = np.argpartition(eng_rf_features, -10)[-10:]

rf_importance = X.columns[top10_indices]
rf_features = eng_rf_features[top10_indices]


# In[110]:


eng_rf_features = pd.DataFrame({"Feature":rf_importance,"Importance":rf_features})
eng_rf_features = eng_rf_features.sort_values("Importance", ascending = False)

eng_rf_features


# In[114]:


sns.barplot(data=eng_rf_features, x="Importance", y='Feature', orient='h')

plt.title("Random Forest: Feature Importances for Employee Leaving", fontsize=12)
plt.ylabel("Feature")
plt.xlabel("Importance")

plt.show()


# As for the decision tree model, the random forest model indicates that the most important features for defining if an employee will leave or will stay are:
# last evaluation, number of projects, tenure and overworked in this order.

# # PACE: Execute Stage
# - Interpret model performance and results
# - Share actionable steps with stakeholders
# 
# 

# ✏
# ## Recall evaluation metrics
# 
# - **AUC** is the area under the ROC curve; it's also considered the probability that the model ranks a random positive example more highly than a random negative example.
# - **Precision** measures the proportion of data points predicted as True that are actually True, in other words, the proportion of positive predictions that are true positives.
# - **Recall** measures the proportion of data points that are predicted as True, out of all the data points that are actually True. In other words, it measures the proportion of positives that are correctly classified.
# - **Accuracy** measures the proportion of data points that are correctly classified.
# - **F1-score** is an aggregation of precision and recall.
# 
# 
# 
# 
# 

# 💭
# ### Reflect on these questions as you complete the executing stage.
# 
# - What key insights emerged from your model(s)?
# - What business recommendations do you propose based on the models built?
# - What potential recommendations would you make to your manager/company?
# - Do you think your model could be improved? Why or why not? How?
# - Given what you know about the data and the models you were using, what other questions could you address for the team?
# - What resources do you find yourself using as you complete this stage? (Make sure to include the links.)
# - Do you have any ethical considerations in this stage?
# 
# 

# ## Step 4. Results and Evaluation
# - Interpret model
# - Evaluate model performance using metrics
# - Prepare results, visualizations, and actionable steps to share with stakeholders
# 
# 
# 

# In[115]:


print(all_results)


# ### Summary of model results
# 
# The **logistic regression** model yielded the following results:
# - Precision: 79%
# - Recall: 82%
# - F1-Score: 80%
# - Accuracy: 82%
# 
# The **decision tree** model on the engineered data set achieved following results:
# - Precision: 86%
# - Recall: 90%
# - F1-Score: 88%
# - Accuracy: 96%
# 
# The random forest model outperforms slightly the decision tree model.

# ### Conclusion, Recommendations, Next Steps
# 
# The data set analysis and constructed models showed that the most of the employees are overworked.
# 
# The following steps could be taken to retain employees:
# - An employees should work on maximum 4 projects.
# - The reasons behind drastic employee dissatisfaction at 4-year tenure should be looked at.
# - Longer working hours should be not a requirement, or should be compensated accordingly.
# - The workload and overtime company policies should be clear to the employees. Employees should be aware about compensations and rewards for taking more projects and making more hours.
# - The company values, culture and promotion requirements should be made clear on the company-wide and inter-group levels.
# - The high evaluation score should be given only to people working long hours.
# 
# Next steps for the data analysis:
# - Exclude *'last_evaluation'* from the data set. This variable can be subjective or outdated
# - K-means modelling can reveal clusters within the data that identify certain employee groups that are more probable to leave the company
