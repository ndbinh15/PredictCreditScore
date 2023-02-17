# PredictCreditScore (Machine Learning)
> [link to this dataset](https://www.kaggle.com/datasets/parisrohan/credit-score-classification?datasetId=2289007)

>Content:
>
>1.1.	Overview dataset
>
>1.2.	Steps to build machine learning model
>
>1.3.	Edit columns and Data Type
>
>1.4.	Find missing data
>
>1.5.	Visualization
>
>1.5.1.	Kernel density estimate (KDE):
>
>1.5.2.	Box plot (or box-and-whisker plot):
>
>1.6.	Detect Outliers and Fill NaN Values
>
>1.7.	Some things to do before building the model
>
>1.8.	Concepts of some measure in the document
>
>1.8.1.	Confusion Matrix:
>
>1.8.2.	Precision:
>
>1.8.3.	Recall:
>
>1.8.4.	F1-score:
>
>1.9.	Modeling:
>
>1.9.1.	NaiveBayers
>
>1.9.2.	KNN
>
>1.9.3.	Decision Tree
>
>1.9.4.	Random forest
>
>1.9.5.	SVM(Support Vector Machines)
>
>#### REFERENCE

## Overview dataset 

> <b>ID:</b> &ensp;	Represents a unique identification of an entry 
>
> <b>Customer ID:</b> &ensp;	Represents a unique identification of a person 
>
> <b>Month:</b> &ensp;	Represents the month of the year 
>
> <b>Name:</b> &ensp;	Represents the name of a person 
>
><b>Age:</b> &ensp;	Represents the age of the person 
>
><b>SSN:</b> &ensp;	Represents the social security number of a person 
>
><b>Occupation:</b> &ensp;	Represents the occupation of the person 
>
><b>Annual_Income:</b> &ensp;	Represents the annual income of the person 
>
><b>Monthly_Inhand_Salary:</b> &ensp;	Represents the monthly base salary of a person 
>
><b>Num_Bank_Accounts:</b> &ensp;	Represents the number of bank accounts a person holds 
>
><b>Num_Credit_Card:</b> &ensp;	Represents the number of other credit cards held by a person 
>
><b>Interest_Rate:</b> &ensp;	Represents the interest rate on credit card 
>
><b>Num_of_Loan:</b> &ensp;	Represents the number of loans taken from the bank 
>
><b>Type_of_Loan:</b> &ensp;	Represents the types of loan taken by a person 
>
><b>Delay_from_due_date:</b> &ensp;	Represents the average number of days delayed from the payment date 
>
><b>Num_of_Delayed_Payment:</b> &ensp; Represents the average number of payments delayed by a person 
>
><b>Changed_Credit_Limit:</b> &ensp;	Represents the percentage change in credit card limit 
>
><b>Num_Credit_Inquiries:</b> &ensp;	Represents the number of credit card inquiries 
>
><b>Credit_Mix:</b> &ensp;	Represents the classification of the mix of credits 
>
><b>Outstanding_Debt:</b> &ensp;	Represents the remaining debt to be paid (in USD) 
>
><b>Credit_Utilization_Ratio:</b> &ensp;	Represents the utilization ratio of credit card 
>
><b>Credit_History_Age:</b> &ensp;	Represents the age of credit history of the person 
>
><b>Payment_of_Min_Amount:</b> &ensp; Represents whether only the minimum amount was paid by the person 
>
><b>Total_EMI_per_month:</b> &ensp;	Represents the Equated Monthly Installments payments (in USD) 
>
><b>Amount_invested_monthly:</b> &ensp; Represents the monthly amount invested by the customer (in USD) 
>
><b>Payment_Behaviour:</b> &ensp;	Represents the payment behavior of the customer (in USD) 
>
><b>Monthly_Balance:</b> &ensp;	Represents the monthly balance amount of the customer (in USD) 
>
><b>Credit_Score:</b> &ensp;	Represents the bracket of credit score (Poor, Standard, Good) 
>
><b>Data types:</b> &ensp; String, Decimal, Integer and Categorical 

![image](https://user-images.githubusercontent.com/58379925/219735840-c06e20bd-3892-40e2-b354-1f8693c8b8cf.png)

> Figure 1: Some noisy and missing of the dataset 

This dataset has 100000 rows and 28 columns, we use the information above to find the credit score. When we look around the dataset, it has many noisy and missing data, so before we build models, we need to detect it and fill NAN values 

### 1.2.	Steps to build machine learning model
- 	Importing the libraries
-	Reading and show the data
-	Edit columns and Data Type
-	Find missing data
-	Detect Outliers and Fill NaN Values for Every columns
-	Save process DATA to CSV
-	Drop unimportant columns
-	Encoding categorical features
-	Scaling and Split the data
-	Model * 5
-	NaiveBayers
-	KNN
-	Decision Tree
-	Random forest
-	SVM(Support Vector Machines)

### 1.3.	Edit columns and Data Type
In the table, we checked the data types in the dataset, it has just one data type (object). So, we need to convert it to some main data types such as integer, float, and object<br>

Almost all rows are the same way to convert. Just ‘Credit_History_Age’ row is more different because we need to split years and months (object) to months (float). So, we built a ‘Month_Convert’ method to do this task.<br>


### 1.4.	Find missing data
Next step is to find missing data. In this section, we need to know where the columns missing are? - we rely on the data types which are distributed in the columns in section before such as “object”, “int” and “float” to 2 groups to easy to observe<br>

### 1.5.	Visualization
Before we go to step fill NAN, we need to define some graph we use in the documentation:<br>

We use the library “Seaborn” to do the task. We take graph kernel density estimate (KDE) and box plot (or box-and-whisker plot) to describe how the datas in each column show up.<br>

#### 1.5.1.	Kernel density estimate (KDE):
A kernel density estimate (KDE) plot is a method for visualizing the distribution of observations in a dataset, analogous to a histogram and the Probability Density of a continuous variable. It depicts the probability density at different values in a continuous variable. We can also plot a single graph for multiple samples which helps in more efficient data visualization.<br>
The concept of weighting the distances of our observations from a particular point, x, can be expressed mathematically as follows:<br>

The variable K represents the kernel function. Using different kernel functions will produce different estimates. The kernel function consisting of Epanechnikov, Normal, Uniform and Triangular<br>
The Kernel Density Estimation works by plotting out the data and beginning to create a curve of the distribution. The curve is calculated by weighing the distance of all the points in each specific location along the distribution.<br>

More information: Kernel Density Estimation (mathisonian.github.io) <br>

#### 1.5.2.	Box plot (or box-and-whisker plot):
A box plot (or box-and-whisker plot) shows the distribution of quantitative data in a way that facilitates comparisons between variables or across levels of a categorical variable. The box shows the quartiles of the dataset while the whiskers extend to show the rest of the distribution, except for points that are determined to be “outliers” using a method that is a function of the inter-quartile range.<br>
A box and whisker plot, often known as a box plot, shows a five-number summary of a collection of data. The lowest, first quartile, median, third quartile, and maximum are the five-number summary.<br>
A box plot is created by drawing a box from the first to third quartiles. At the median, a vertical line runs through the box. The whiskers move from the lowest to the highest quartile.<br>

More information: Box plot review (article) | Khan Academy<br>

### 1.6.	Detect Outliers and Fill NaN Values
We will illustrate some sample columns of how we preprocess the data:<br>

The graphs above showed us information about data allotment in the “Monthly_Inhand_Salary” column. Mostly data located on ~0 - 12500 and it went down moderately on about 15000. On 12500 - 15000, data allotment was lower than others, called outliers. So, we need to split some of them to minimize the outlier data which can skew results and anomalies in training data can impact overall model effectiveness. Purpose to safeguarding data quality <br>

In this step, first, we must check where rows on the column are null and remove missing values. Try to fill Monthly_Inhand_Salary[0] value, either fill NaN value. When the row is not null, take the value of it to use this in the next loop.<br>

The outliers were lower than after we deleted outliers. So, we are finished to detect outliers and fill NAN to the “Monthly_Inhand_Salary” column.<br>

Why did not clear all outliers? - Because they are the part of our data, we just need to minimize them, and keep them will be more interesting our dataset but not too affected to the data quality<br>

In other columns we use the same way to detect outliers and delete them. So, we just show how the graphs change before and after detecting outliers and deleting them.<br>


### 1.7.	Some things to do before building the model
We need to drop some unimportant columns because we do not use their columns to build models, it’s unnecessary. Another thing is encoding categorical features, we change them to integers following their order. So that the data with converted categorical values can be provided to the models to give and improve the predictions<br>

### 1.8.	Concepts of some measure in the document
The last one, before going to build models machine learning step, we need to know some measure to check model performance.<br>
#### 1.8.1.	Confusion Matrix: 
Helps us to display the performance of a model or how a model has made its prediction
in Machine Learning.

-	True Positive: The number of times a model properly identifies a positive sample as Positive.
-	False Negative: How many times does a model wrongly classify a positive sample as Negative?
-	False Positive: How many times does a model mistakenly classify a negative sample as positive?
-	True Negative: How many times does a model accurately classify a negative sample as Negative?
#### 1.8.2.	Precision: 
Defined as the ratio of correctly classified positive samples (True Positive) to a total number of classified positive samples (either correctly or incorrectly). Examine the data set to see how much of it the model properly anticipated. This is the accuracy stat.<br>
Precision = True Positive/True Positive + False Positive <br>
#### 1.8.3.	Recall: 
Determined as the proportion of Positive samples that were properly identified as Positive to the total number of Positive samples. The recall of the model assesses its ability to recognize positive samples. The more positive samples identified, the larger the recall. This statistic is also known as coverage, and it considers how effectively the final model is generalizable.<br>
Recall = True Positive/True Positive + False Negative<br>
Recall, unlike Precision, is unaffected by the amount of negative sample classifications. Furthermore, if the model identifies all positive samples as positive, Recall equals one.<br>

#### 1.8.4.	F1-score: 
From the two factors of accuracy and coverage, another index is given called F1-Score. This is known as a harmonic mean of the Precision and Recall. It tends to take the value closest to the lesser of the Precision and Recall values, and it has a big value if both the Precision and Recall values are large. As a result, F1-Score provides a more objective representation of the performance of a machine learning model.<br>

### 1.9.	Modeling:
#### 1.9.1.	NaiveBayers

This is a technique that has remained popular over the years. <br>
Naive Bayes classification is an easy-to-use and effective technique for the classification task in machine learning. The foundation of naive Bayes classification is the use of the Bayes theorem with a strong assumption of feature independence. When we apply naive Bayes classification to textual data analysis tasks like natural language processing, we get good outcomes.<br>
Naive Bayes Classifier uses the Bayes’ theorem, the class with the highest probability is considered as the most likely class. This is also known as the Maximum A Posteriori (MAP). <br>
> MAP (A)
>
> = max (P (A | B))
>
> = max (P (B | A) * P (A))/P (B)
>
> = max (P (B | A) * P (A))

Nave Bayes Classifier assumes that there is no relationship between any two features. A feature's presence or absence has no bearing on whether another feature is present or not.<br>
In real-world datasets with many pieces of feature-based evidence. Consequently, the calculations become very challenging. The feature independence strategy is used to decouple several pieces of evidence and treat each as a separate piece in order to streamline the task.<br>

#### 1.9.2.	KNN
One of the simplest machine learning algorithms, based on the supervised learning method, is K-Nearest Neighbour.<br>
Since K-NN is a non-parametric technique, it makes no assumptions about the underlying data.<br>
It is also known as a lazy learner algorithm since it saves the training dataset rather than learning from it immediately. Instead, it uses the dataset to perform an action when classifying data.<br>
K-NN working:<br>
✔	Step 1: Decide on the neighbors' K-numbers.<br>
✔	Step 2: Determine the Euclidean distance between K neighbors.<br>
✔	Step 3: Based on the determined Euclidean distance, select the K closest neighbors.<br>
✔	Step 4: Count the number of data points in each category among these k neighbors.<br>
✔	Step 5: Assign the fresh data points to the category where the neighbor count is highest.<br>
✔	Step 6: Our model is complete.<br>

#### 1.9.3.	Decision Tree
The Decision Tree is a Supervised learning approach that may be used to solve both classification and regression issues, however it is most employed to solve classification problems. It is a tree-structured classifier, with core nodes representing dataset attributes, branches representing decision rules, and leaf nodes representing outcomes.<br>
The Decision Node and the Leaf Node are the two nodes of a Decision tree. Decision nodes are used to make any decision and have several branches, whereas Leaf nodes are the results of such decisions and have no additional branches.<br>
A decision tree simply asks a question, and then divides the tree into subtrees based on the answer (Yes/No).<br>
Decision Tree algorithm works:<br>
✔	Step 1: Begin the tree with the root node, which includes the whole dataset, explains S.<br>
✔	Step 2: Using the Attribute Selection Measure, find the best attribute in the dataset (ASM).<br>
✔	Step 3: Subdivide the S into subsets containing potential values for the best qualities.<br>
✔	Step 4: Create the decision tree node with the best attribute.<br>
✔	Step 5: Create new decision trees recursively using the subsets of the dataset obtained in step 3. Continue this procedure until you reach a point where you can no longer categorize the nodes and refer to the last node as a leaf node.<br>

#### 1.9.4.	Random forest 
Random Forest is a well-known machine learning algorithm from the supervised learning approach. It is built on the notion of ensemble learning, which is a method that involves integrating several classifiers to solve a complicated issue and enhance the model's performance.<br>
Random Forest is a classifier that uses a number of decision trees on different subsets of a given dataset and averages them to enhance the predicted accuracy of that dataset. Instead of depending on a single decision tree, the random forest collects the forecasts from each tree and predicts the final output based on the majority vote of predictions.<br>
The bigger the number of trees in the forest, the higher the accuracy and the lower the risk of overfitting.<br>

The following steps describe the working process:<br>
✔	Step 1: Choose K data points at random from the training set.<br>
✔	Step 2: Create decision trees for the specified data points (Subsets).<br>
✔	Step 3: Determine the number N for the number of decision trees you wish to construct.<br>
✔	Step 4: Repeat Steps 1 and 2.<br>
✔	Step 5: Find the forecasts of each decision tree for new data points and allocate the new data points to the category with the most votes.<br>

#### 1.9.5.	SVM(Support Vector Machines)
Support Vector Machine” (SVM) is a supervised machine learning algorithm that can be used for both classification and regression challenges. However,  it is mostly used in classification problems<br>
The objective of the support vector machine algorithm is  to choose the best line to classify your data points. It chooses the line that separates the data and is the furthest away from the closest data points as possible.<br>
There are two kinds of SVM:<br>
-	Linear SVM is used for linearly separable data, which implies that if a dataset can be categorized into two classes using a single straight line, it is considered linearly separable data, and the classifier employed is the Linear SVM classifier.
-	Non-linear SVM is used for non-linearly separated data, which implies that if a dataset cannot be categorized using a straight line, it is considered non-linear data, and the classifier employed is the Non-linear SVM classifier.
We will use the Linear SVM to build our model.<br>

# REFERENCE
[seaborn.boxplot — seaborn 0.12.0 documentation (pydata.org) ](https://seaborn.pydata.org/generated/seaborn.boxplot.html)

[seaborn.kdeplot — seaborn 0.12.0 documentation (pydata.org) ](https://seaborn.pydata.org/generated/seaborn.kdeplot.html)

[Naive Bayes Classifier in Machine Learning - Javatpoint ](https://www.javatpoint.com/machine-learning-naive-bayes-classifier)

[K-Nearest Neighbor(KNN) Algorithm for Machine Learning - Javatpoint ](https://www.javatpoint.com/k-nearest-neighbor-algorithm-for-machine-learning)

[Machine Learning Decision Tree Classification Algorithm - Javatpoint ](https://www.javatpoint.com/machine-learning-decision-tree-classification-algorithm)

[Support Vector Machine (SVM) Algorithm - Javatpoint ](https://www.javatpoint.com/machine-learning-support-vector-machine-algorithm)

[Machine Learning Random Forest Algorithm - Javatpoint ](https://www.javatpoint.com/machine-learning-random-forest-algorithm)

[Naive Bayes Classifier in Python | Kaggle ](https://www.kaggle.com/code/prashant111/naive-bayes-classifier-in-python)

[A Guide to any Classification Problem | Kaggle ](https://www.kaggle.com/code/durgancegaur/a-guide-to-any-classification-problem)

[Credit score classification 80% score 7 models | Kaggle ](https://www.kaggle.com/code/abdelaziznabil/credit-score-classification-80-score-7-models)

[Credit Score Classification Cleaning,EDA,Modling | Kaggle ](https://www.kaggle.com/code/mohamedramadanyakoub/credit-score-classification-cleaning-eda-modling)

[SVM: Difference between Linear and Non-Linear Models - AITUDE ](https://www.aitude.com/svm-difference-between-linear-and-non-linear-models/)

[Micro, Macro & Weighted Averages of F1 Score, Clearly Explained | by Kenneth Leung | Towards Data Science ](https://towardsdatascience.com/micro-macro-weighted-averages-of-f1-score-clearly-explained-b603420b292f)

[King_County_House_Prices (franciscadias.com) ](https://franciscadias.com/7_King_County_House_Prices.html)

[Giới thiệu về k-fold cross-validation - Trí tuệ nhân tạo (trituenhantao.io) ](https://trituenhantao.io/kien-thuc/gioi-thieu-ve-k-fold-cross-validation/)

[Một vài hiểu nhầm khi mới học Machine Learning (viblo.asia) ](https://viblo.asia/p/mot-vai-hieu-nham-khi-moi-hoc-machine-learning-4dbZNoDnlYM)

> This assignment is created by lecturer: Assoc.Prof.PhD. Le Anh Cuong
> 
> Head - Department of Computer Science
> 
> Director - NLP-KD Laboratory Ton Duc Thang University

> I'm finished this assignment at Ton Duc Thang University - Year: 2022
> 
> But it is not perfect so there are many problems. Be careful if you take it
