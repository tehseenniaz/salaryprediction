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
This is the distribution of salary data
![](https://github.com/tehseenniaz/salarypredictionportfolio/blob/master/1.png)

It's a typical salary distribution with right-skew representing very high earners. I could log-transform it for better symmetry but did not for two reasons:
1. The distribution is generally symmetric so it's not a big risk.
2. Saying our model is generally off by X thousand dollars is far easier to interpret for stakeholders than saying it is off by X log dollars.
