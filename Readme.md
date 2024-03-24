## Salifort Motors: Employee Retention Project

#### Problem:

Salifort Motors seeks to improve employee retention and answer the following question:

**Why do the employees leave the company?**

#### Approach:

Since the variable we are seeking to predict is categorical, we could build either a logistic regression or a tree-based machine learning model.

The models will help predict whether an employee will leave and identify which factors are most influential. These insights can help to come to conclusions on how to improve employee retention.

#### Results:

Decision tree model concluded that the most relevant variables are ‘last_evaluation’, ‘number_project’, ‘tenure’ and ‘overworked’.

The random forest model showed that `last_evaluation`, `tenure`, `number_project`, `overworked`, `salary_low`, and `work_accident` have the highest importance. These variables are most helpful in predicting the outcome variable, `left`.
