# The Salary Estimation Problem
Determining the right salary for current and prospective employees is always a bit tricky. Offering too low can turn away prospective and current talent; offering too much can make the business less competitive with not enough incentive for employees to work hard for better rewards. 

Most people look up average or median salaries for any job title and use that as an anchor. However, accounting for multiple other factors and the nature of their interactions can help us become much more objective, precise, and equitable for all parties. 

This data contains features such as job title, education, industry, experience, and distance from metropolis to predict any given candidate's salary.
There's also a salaries dataset that contains the corresponding salaries, as well as a "testing" dataset to deploy our final model on and make our predictions.

## Data Cleaning
For data cleaning:
- I checked if there were any duplicates and null values and there were none.
- The salary data had 5 rows with 0 salary and those were dropped.
- I also dropped the companyId column because I wanted to generalize the predictions beyond the 63 companies and to avoid explosion of dimensionality and sparsity of the data when creating dummies.

Finally, I split the data into 75% training and 25% testing sets.

# Exploratory Data Analysis

### Distribution of salary data

![](https://github.com/tehseenniaz/salarypredictionportfolio/blob/master/1.png)

It's a typical salary distribution with right-skew representing very high earners. I could log-transform it for better symmetry but did not for two reasons:
1. The distribution is generally symmetric so it's not a big risk.
2. Saying our model is generally off by X thousand dollars is far easier to interpret for stakeholders than saying it is off by X log dollars.

Now, we evaluate each categorical feature against salary and see if the categories differ. Here are two examples of Salary by job type and degree.

### Salary by Job type

![](https://github.com/tehseenniaz/salarypredictionportfolio/blob/master/2.png)

### Salary by degree

![](https://github.com/tehseenniaz/salarypredictionportfolio/blob/master/3.png)

This shows that there's notable variation in salary by each group, and so they have some predictor power and should be used in our models.  
After categorical features, now we compare numeric features against salary.
#### Correlation plot
![](https://github.com/tehseenniaz/salarypredictionportfolio/blob/master/4.png)
#### Correlation Heatmap
![](https://github.com/tehseenniaz/salarypredictionportfolio/blob/master/hm.png)

Salary is higher with more experience.  
Salary is lower as distance from metropolis increases.  
Experience and distance have almost no correlation.

# Establish a baseline
We will predict salary using industry median to establish a baseline prediction method for comparison.  
We need to do better than MSE of 1624, which is like making an error of about 40k USD on average per prediction.

# Hypothesize Solution
Capture linear associations between variables and target:
- Linear Regression
- Linear Regression with interactions

Capture non-linear relations in the feature space:
- Random Forest - regression trees with bagging and sampled features
- Light GBM - leaf-wise splitter that is a lot faster than most other boosting methods with comparable or better performance

Combination:
- Stacking - stack two or more models to collectively predict salary

# Feature Engineering
For feature engineering, I:
1. Created dummy variables for all categories and then dropped one subgroup from each categorical variable for appropriate regression inputs, otherwise there's perfect multicollinearity.

2. Created interactions for each variable with all other variables. Here are justifications for four of fifteen interactions:
- JobType * Industry  
CFO in finance industry would earn more than a CFO in education, which means that the effect of job type on salary depends on industry, so they interact.

- JobType * Degree --- | --- JobType * Major  
A CTO with literature major ought to earn differently from a CTO with computer science major.

- JobType * yearsExperience  
25 years of experience as a janitor ought to be different from same experience as a senior, so salary changes due to experience differs by job type.  

This results in 378 features if we include all interactions terms, which is a lot. We could do principal component analysis for reducing this deminsionality but I didn't because:
1. PCA doesn't work well with highly sparse data
2. We have about 2000 observations per feature as it is

# Create and evaluate models
All models were evaluated with at least 3-fold cross-validation with their average MSE and standard deviations reported.
### Model 1: Linear Regression
Avg_score: 384.73  
Std.dev: 1.064
### Model 2: Linear Regression with interactions
Avg_score: 354.17  
Std.dev: 0.767
### Models 3, 4, and 5: [ Random Forest, LightGBM, Gradient Boosting ]
I did randomized grid search to get quasi-optimal hyperparameters on the dummy_train data set. Since this is computationally very intensive, I used Google Colaboratory's fastest processors, Tensor Processing Units (TPUs), to get the tuned hyperparameters. The final results from Google Colab are printed below.

- model: lgbm (LightGBM)  
Avg cv-score with best parameters: 357.68  
best parameters: {'num_leaves': 63, 'max_depth': 12}  

- model: rfr (Random Forest Regressor)  
Avg cv-score with best parameters: 444.17  
best parameters: {'n_estimators': 150, 'max_features': 'sqrt'}  

- model: gbr (Gradient Boosting Regressor)  
Avg cv-score with best parameters: 397.17  
best parameters: {'subsample': 0.1, 'n_estimators': 130, 'max_features': 'auto', 'max_depth': 16, 'learning_rate': 0.05}  

### Model 6: Stacking
I used the StackingRegressor using a combination of our best performing and fastest linear and tree-based methods - linear regression with interactions and LightGBM, respectively. The interactions data was passed through to the StackingRegressor along with the combined predictions of these two models to make final predictions.  

The results on 3-fold cross-validation:  
Avg_score: 354.2  

### Testing models on unseen data
I select the stacked model based on its performance on cross-validated data where the test set should only be used for gauging model performance as opposed to model selection. However, I still want to see how each model performs on the test set out curiosity - not for final selection which we have already made. Here's how they performed:

![](https://github.com/tehseenniaz/salarypredictionportfolio/blob/master/6.PNG)

The linear model with and without interactions and the stacked model differ by mere decimal points around 353 on the testing set. We have already decided on the stacked model so this just confirms its stable performance since its almost identical with the cross-validated score.  

Stacked model's performance on test set is **MSE of 353.53**, so we are making an error of about **USD 18,800** on average per prediction.  
This is a significant improvement over using industry average with **MSE of 1624** or an error of about **USD 40,200** on average per prediction.  

Since there's no way to plot feature importances from a stacking regressor, we will see feature importance from LightGBM which was in the stacked model.  
![](https://github.com/tehseenniaz/salarypredictionportfolio/blob/master/7.PNG)

It appears that where a person lives is the most important predictor after controlling for all other factors and their influences on each other at each level of interaction.   

This is a somewhat surprising revelation, but it could speak to the effect of non-professional socio-economic factors in determining financial success and everything that leads up to it.  

Experience, whether they are in the finance or oil industry, with or without an engineering background, etc. are also some of the most potent predictors of salary.
