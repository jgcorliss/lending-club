# Predicting Loan Defaults on LendingClub.com
[LendingClub](https://www.lendingclub.com/) is a US peer-to-peer lending company and the world's largest peer-to-peer lending platform. In this project, I build machine learning models to predict the probability that a loan on LendingClub will charge off (default). These models could help LendingClub investors make better-informed investment decisions. I use a 1.8 GB LendingClub [dataset](https://www.kaggle.com/wordsforthewise/lending-club) with 1,646,801 loans and 150 variables for each loan.

In training the models, I only use features that are known to investors before they choose to invest in the loan. These features include, among others, the borrower's income, FICO score, and debt-to-income ratio, and the loan amount, purpose, grade, and interest rate.

The modeling process takes several steps, including: removing loan features with significant missing data, or that aren't known to investors; exploring, transforming, and visualizing the data; creating dummy variables for categorical features; and fitting three models: logistic regression, random forest, and k-nearest neighbors. I use machine learning pipelines to combine imputation, standardization, dimension reduction, and model fitting into one pipeline object. I optimize hyperparameters through a cross-validated grid search.

I found that the three models performed similarly well according to cross-validated AUROC scores on the training data. I chose logistic regression as the final model, which obtained an AUROC score of 0.689 on a test set consisting of the most recent 10% of the loans.

I also found that, according to Pearson correlations, the most useful variables for predicting charge-off are the loan interest rate, the loan term, the borrower's FICO score, and the borrower's debt-to-income ratio.

All the analysis is done in a Python Jupyter Notebook, utilizing the packages numpy, pandas, matplotlib, seaborn, and scikit-learn.
