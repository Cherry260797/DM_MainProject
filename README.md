# DM_MainProject
# Deployment Instruction
Open the Python Note Book with Google Colab or Jupyter NoteBook.
Execute the cells in the order in order to get each cell's output.
Python Version used is 3.10.2
Also make sure tha the following library related things are installed using 'pip install'
# Import libraries for manipulating numpy arrays and pandas data frames
import pandas as pd
import numpy as np

# Import libraries for word, string, regular expression related operations
import re
import string
from collections import defaultdict
from wordcloud import WordCloud,STOPWORDS

# Import libraries for plots
import matplotlib.pyplot as plt
import seaborn as sns
from plotly import tools
from plotly import subplots
import plotly.graph_objs as go
from plotly.offline import iplot

# Import nltk libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Import libraries for splitting dataset and implementing algorithms, calculating performances
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

import time



# Explanation
Introduction :
Amazon and Flipkart have become the most frequent e-commerce websites today that millions of their products and reviews play a very vital role in data analysis.  Dataset from Kaggle at https://www.kaggle.com/datasets/vivekgediya/ecommerce-product-review-data?select=Flipkart_Reviews+-+Electronics.csv is considered for analysis. Using this dataset, analysis of amazon and flipkart product reviews is carried out using different Machine Learning classification algorithms. This blog contains the step wise procedure followed for executing different classification algorithms for analysis.

Please Find the code for this blog here.

Please find a brief YouTube video here.

Algorithmic Explanation :
Classification Algorithms are used for categorizing the given data into classes or categories. Let's look at some classification algorithms.


1. Logistic Regression : 
Logistic Regression algorithm applies the logistic function on the linear regression which shows the relation between the dependent features/variables and independent classes/categories/variables. Thus firstly this relation is found by calculating all the required parameters and then the class boundary is found applying the logistic function. This class boundary helps us categorize the given data point.

      Pros :

        1. Simple and effective to implement.

        2. Feature scaling and hyper parameter tuning aren't required.

      Cons :

        1. It has high dependency on proper presentation of data.

        2. Low performance on non linear data.

        3. Low performance on data with correlated features.

2. KNN :
In the K Nearest Neighbor algorithm, distance of the test data point from all the training data points(with labels) is calculated. Based on the type of the data, this calculated distance can be of the form Euclidean, Manhattan, Minkowski, Jaccard distance etc. Thus 'n' nearest neighbors with least distance from the test data point are collected and the label existing with the maximum number of neighbors is the predicted label or class.

      Pros :

        1. Multi class problems can be solved using it.

        2. It's a simple, constantly evolving model without any assumptions about data.

      Cons :

        1. Slow for large datasets.

        2. Doesn't work well with high dimensional data.

        3. Sensitive to outliers.

        4. Can't deal with missing values.

3. SVM :
In Support Vector Machine algorithm, firstly, support vectors which are data points  belonging to different existing classes with the least distance between them are found out. Maximizing the marginal distance between them, class boundary is found out using which the test data point can be categorized into any of the existing classes.

       Pros : 

         1. It works well in higher dimensional data. 

         2. Outliers in SVM has less impact.

         3. It works best when classes are separable.

      Cons :

         1. It's slow and time consuming.

         2. It has poor performance with overlapped classes.

         3. Selection of hyper parameters and kernel function is important.

4. Naive Bayes :
In the Naive Bayes Algorithm, Probability that a given test data point belongs to a particular class is calculated using Bayes Theorem, Conditional Probability concepts. Thus the data point is predicted to be belonging to the class/category with the highest probability.

         Pros :

           1. It's very fast real time prediction algorithm.

           2. It performs well with high dimensional and multi class data also.

           3. It's insensitive to irrelevant features.

        Cons :

           1. It requires training data to represent population well.

           2. Naive Assumption.

5. Decision Trees :
In the Decision Trees algorithm, the features or attributes with their values are used to calculate the entities of gini index or entropy based on which the information gain provided by each attribute is calculated. Decision Tree is built by splitting with respect to the attributes providing the highest information gain.

         Pros :

           1. Automatic Feature Selection.

           2. Easy explanation and visualization.

           3. Normalization or scaling of data isn't needed.

         Cons :

           1. Prone to overfitting, sensitive to data.

           2. Higher time required to train decision trees.

6. Bagging Classifier :
A Bagging classifier is an ensemble meta-estimator that fits base classifiers each on random subsets of the original dataset and then aggregate their individual predictions (either by voting or by averaging) to form a final prediction. Such a meta-estimator can typically be used as a way to reduce the variance of a black-box estimator (e.g., a decision tree), by introducing randomization into its construction procedure and then making an ensemble out of it.

7. Random Forest Classifier :
A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.

8. Gradient Boosting Classifier :
This algorithm builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions. In each stage n_classes_ regression trees are fit on the negative gradient of the loss function, e.g. binary or multiclass log loss. Binary classification is a special case where only a single regression tree is induced.

9. Ridge Classifier :
It works by adding a penalty term to the cost function that discourages complexity. The penalty term is typically the sum of the squared coefficients of the features in the model. This forces the coefficients to remain small, which prevents overfitting. The amount of regularization can be controlled by changing the penalty term. A larger penalty results in more regularization and a smaller coefficient values. This can be beneficial when there is little training data available. However, if the penalty term is too large, it can result in underfitting.


Data Set :
From the Kaggle dataset two files are considered, "Product Review Large Data.csv" and "Flipkart_Reviews - Electronics.csv."

These files include reviews in the form of ratings, textual review, helpfulness votes and the product details of it's ID, category, price, brand and few other details of time and place of review, product usage.


"Flipkart_Reviews - Electronics.csv." file has 9374 rows and 9 columns.

"Product Review Large Data.csv" file has 10971 rows and 27 columns.

Procedural Explanation :
This analysis is carried out in the following steps

Importing Libraries

Loading the Dataset

Data Preprocessing, Exploratory Data Analysis and Visualization

Building the Models

Comparing their performance

1. Importing the Libraries 
Import all the libraries required for execution of the required machine learning algorithms.


# Import libraries for manipulating numpy arrays and pandas data frames
import pandas as pd
import numpy as np

# Import libraries for word, string, regular expression related operations
import re
import string
from collections import defaultdict
from wordcloud import WordCloud,STOPWORDS

# Import libraries for plots
import matplotlib.pyplot as plt
import seaborn as sns
from plotly import tools
from plotly import subplots
import plotly.graph_objs as go
from plotly.offline import iplot

# Import nltk libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Import libraries for splitting dataset and implementing algorithms, calculating performances
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

import time
2. Loading the Dataset
 Loading the Data from "Flipkart_Reviews - Electronics.csv" using pandas library, first few rows of the data frame are printed and checked.

flipkart_dataframe = pd.read_csv("/kaggle/input/ecommerce-product-review-data/Flipkart_Reviews - Electronics.csv")
flipkart_dataframe.head()

Product_id, product_title represent the metadata of the products.

Rating, Summary, Review, Upvotes and Downvotes represent the reviews related stuff.

Date and location of the review represent the time and location of review.

Loading the Data from "Product Review Large Data.csv" using pandas library, first few rows of the data frame are printed and checked.

flipkartamazon_df = pd.read_csv("/kaggle/input/ecommerce-product-review-data/Product Review Large Data.csv")
flipkartamazon_df.head()
Check the dimensions of the data frames

print(flipkartamazon_df.shape)
print(flipkart_dataframe.shape)
       Size of flipkart_dataframe is (9374, 9).

       Size of flipkartamazon_df is (10971, 27).

3. Data Preprocessing, Exploratory Data Analysis and Visualization
Checking the information about flipkart_dataframe.

flipkart_dataframe.info()











    It is observed that all the attribute values are non null and there are no missing values.

Checking the information about the flipkartamazon_df

flipkartamazon_df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10971 entries, 0 to 10970
Data columns (total 27 columns):
 #   Column                Non-Null Count  Dtype  
---  ------                --------------  -----  
 0   id                    10971 non-null  object 
 1   asins                 1597 non-null   object 
 2   brand                 10971 non-null  object 
 3   categories            10971 non-null  object 
 4   colors                774 non-null    object 
 5   dateAdded             10971 non-null  object 
 6   dateUpdated           10971 non-null  object 
 7   dimension             565 non-null    object 
 8   ean                   898 non-null    float64
 9   keys                  1597 non-null   object 
 10  manufacturer          965 non-null    object 
 11  manufacturerNumber    902 non-null    object 
 12  name                  1597 non-null   object 
 13  prices                1597 non-null   object 
 14  reviews.date          1217 non-null   object 
 15  reviews.doRecommend   539 non-null    object 
 16  reviews.numHelpful    900 non-null    float64
 17  reviews.rating        10551 non-null  float64
 18  reviews.sourceURLs    1597 non-null   object 
 19  reviews.text          10971 non-null  object 
 20  reviews.title         10954 non-null  object 
 21  reviews.userCity      97 non-null     object 
 22  reviews.userProvince  0 non-null      float64
 23  reviews.username      1580 non-null   object 
 24  sizes                 0 non-null      float64
 25  upc                   898 non-null    float64
 26  weight                686 non-null    object 
dtypes: float64(6), object(21)
memory usage: 2.3+ MB
    It shows, there are missing values in this dataframe and they require pre-processing.

flipkartamazon_df.isnull().sum()
id                          0
asins                    9374
brand                       0
categories                  0
colors                  10197
dateAdded                   0
dateUpdated                 0
dimension               10406
ean                     10073
keys                     9374
manufacturer            10006
manufacturerNumber      10069
name                     9374
prices                   9374
reviews.date             9754
reviews.doRecommend     10432
reviews.numHelpful      10071
reviews.rating            420
reviews.sourceURLs       9374
reviews.text                0
reviews.title              17
reviews.userCity        10874
reviews.userProvince    10971
reviews.username         9391
sizes                   10971
upc                     10073
weight                  10285
dtype: int64
Observations :
There are missing values in the large file of amazon and flipkart reviews. They need to be treated.

There are upvotes and downvotes in flipkart reviews file. They can be used to calculate helpfulness and used for analysis.

But there is numHelpful attribute in large file of flipkart and amazon reviews but it's not clear of total votes and proportional helpful votes.

Therefore, helpfulness attribute is ignored for now.

Working on required Attributes / Features
Attribute Consideration
From the flipkart dataframe, only product_id, review, rating and summary are considered as upvotes, downvotes are ignored as stated above.

From flipkartamazon dataframe, id, reviews.txt, reviews.rating, reviews.title attributes are first renamed and considered.

fkdata = flipkart_dataframe[["product_id","review","rating","summary"]]
flipkartamazon_df=flipkartamazon_df.rename(columns={'id':'product_id','reviews.text':'review','reviews.rating':'rating','reviews.title':'summary'})
fkamazondata = flipkartamazon_df[["product_id","review","rating","summary"]]
Treating the missing values
From both the flipkart and flipkartamazon dataframes, data points with the missing ratings and reviews are dropped and ignored.

fkdata = fkdata.dropna()
fkamazondata = fkamazondata.dropna()
Combining the required attributes
From both the dataframes, contents of attributes "review" and "summary" are concatenated and stored under "review" attribute for easy processing.

fkdata['review']=fkdata['review']+fkdata['summary']
fkamazondata['review']=fkamazondata['review']+fkamazondata['summary']
fkdata=fkdata[["product_id","review","rating"]]
fkamazondata=fkamazondata[["product_id","review","rating"]]
Combining both dataframes
Data Frames of both the files are now combined into single dataset.

total_df=[fkdata,fkamazondata]
dataset=pd.concat(total_df)
dataset.head()

Dataset info is verified.

dataset.info()

dataset.describe(include=["O"])

Visulaization
Displaying the most commonly used words in reviews using Word Cloud.

reviewtext = dataset['review']
wordcloud = WordCloud(background_color='white',width=1000,height=400).generate(" ".join(reviewtext))
plt.figure(figsize=(10,5))
plt.imshow(wordcloud)
plt.title('All Words of Reviews\n',size=20)
plt.axis('off')
plt.show()

Displaying the most commonly used  sentiment words from reviews.

A list of most commonly used sentiment words is defined and based on their count in reviews, plotting is done.

words = ['awesome','great','fantastic','extraordinary','amazing','super','magnificent','stunning','impressive','wonderful','breathtaking','love','content','pleased','happy','glad','satisfied','lucky','shocking','cheerful','wow','sad','unhappy','horrible','regret','bad','terrible','annoyed','disappointed','upset','awful','hate']
reviewtext = " ".join(dataset['review'])
dict_words = {}
for word in reviewtext.split(" "):
    if word in words:
        dict_words[word] = dict_words.get(word,0)+1
wordcloud = WordCloud(background_color='white',width=1000,height=400).generate_from_frequencies(dict_words)
plt.figure(figsize=(10,5))
plt.imshow(wordcloud)
plt.title('Sentiment Words\n',size=20)
plt.axis('off')
plt.show()

Visualaizing Ratings and their counts using count plot.

plt.figure(figsize=(10,5))
sns.countplot(dataset['rating'])
plt.title('Count ratings')
plt.show()

Visualizing 'Product ID' Vs 'Mean Rating'

          All the data points are grouped by "product id" and average ratings are calculated for every product. They are plotted as a bar graph.

data_rating = dataset.groupby("product_id").mean().reset_index()
data_rating = data_rating.sort_values(['rating']).reset_index()
plt.figure(figsize=(10,15))
sns.barplot(x=data_rating["rating"], y=data_rating["product_id"])
plt.title('Product ID Vs Mean Rating')
plt.show()




Cleaning Reviews Data
A function to clean the 'review' attribute's data of all data points is defined to convert all it's characters to lower case, remove unnecessary characters like braces, links, punctuations[1]. This function is called and dataset's first few rows are verified.

def clean_reviews(reviewinfo):
    # Convert all the characters to lower case
    reviewinfo = str(reviewinfo).lower()
    # Remove the square braces
    reviewinfo = re.sub('\[.*?\]', '', reviewinfo)
    # Remove the links
    reviewinfo = re.sub('https?://\S+|www\.\S+', '', reviewinfo)
    # Remove the punctuations and number containing words
    reviewinfo = re.sub('<.*?>+', '', reviewinfo)
    reviewinfo = re.sub('[%s]' % re.escape(string.punctuation), '', reviewinfo)
    reviewinfo = re.sub('\n', '', reviewinfo)
    reviewinfo = re.sub('\w*\d\w*', '', reviewinfo)
    return reviewinfo
dataset['review']=dataset['review'].apply(lambda x:clean_reviews(x))
dataset.head()

Tokenization of Words
Tokenization is performed to separate sentences, words, characters, first few rows are printed and verified.

# Words Tokenization
dataset['review'] = dataset.apply(lambda row:word_tokenize(row['review']),axis=1) 
dataset.head()


Removal Of Stop Words
Stop Words that commonly occur in the reviews and that do not require any attention during prediction are removed.

# Remove Stop Words
stop = stopwords.words('english')
dataset['review'] = dataset['review'].apply(lambda x: [item for item in x if item not in stop])
dataset["review"] = dataset["review"].apply(lambda x: str(' '.join(x))) #joining all tokens
dataset.head()

Add an attribute 'Sentiment'
A new attribute 'sentiment' is added to the dataset based on the rating value of each data point. If rating is 1 or 2, sentiment is 0, if rating is 3, sentiment is 1 and if rating is 5, sentiment is 2. Here 0,1,2 resemble negative, neutral and positive reviews respectively.

dict_sentiment = {1: 0,2: 0,3: 1,4: 2,5: 2}
dataset["sentiment"] = dataset["rating"].map(dict_sentiment)
dataset.head()


Plotting the Most Frequent Words of all three classes
All the positive, neutral and negative reviews are grouped separately.

Most frequently occuring words of every class are found out calculating the count of words and plotted as a horizontal bar graph for all the three classes.[1]

# Form three dataframes of positive, negative, neutral reviews
positive_reviews = dataset[dataset["sentiment"]==2]
neutral_reviews = dataset[dataset["sentiment"]==1]
negative_reviews = dataset[dataset["sentiment"]==0]

## Function for Monogram construction ##
def monogram(revtext, monogram=1):
    tok = [tok for tok in revtext.lower().split(" ") if tok != "" if tok not in STOPWORDS]
    ngrams = zip(*[tok[i:] for i in range(monogram)])
    return [" ".join(ngram) for ngram in ngrams]

## Function for horizontal bar chart ##
def hbarchart(datf, color):
    t = go.Bar(
        y=datf["word"].values[::-1],
        x=datf["wordcount"].values[::-1],
        showlegend=False,
        orientation = 'h',
        marker=dict(
            color=color,
        ),
    )
    return t

## Bar Chart of Positive Reviews ##
maxword_dict = defaultdict(int)
for rev in positive_reviews["review"]:
    for term in monogram(rev):
        maxword_dict[term] += 1
sorted_df = pd.DataFrame(sorted(maxword_dict.items(), key=lambda x: x[1])[::-1])
sorted_df.columns = ["word", "wordcount"]
t0 = hbarchart(sorted_df.head(25), 'blue')







4. Building The Models
Building Term Frequency Inverse Document Frequency Matrix[2].

          Inverse Document Frequency measures the proportion of the occurrence of a particular word in the required class or among all the existing reviews.

# Building Term Frequency Inverse Document Frequency Matrix
vectorizer =TfidfVectorizer(max_df=0.9)
text = vectorizer.fit_transform(dataset["review"])
Perform Train Test Split

          Here the dataset is split 75% into the training set and 25% into the test set using train_test_split function[2].

x_train, x_test, y_train, y_test = train_test_split(text, dataset["sentiment"], test_size=0.25, random_state=1)
Models are built with all the classification algorithms mentioned above using sklearn libraries[2].

For every classification algorithm, training and test accuracies are recorded.

For every classification algorithm, execution times are also recorded.

train_accuracy=[]
test_accuracy=[]
exec_time=[]

start_time=time.time()
model_logreg = LogisticRegression(random_state=1,max_iter=200)
model_logreg.fit(x_train, y_train)
y_pred = model_logreg.predict(x_test)
y_pred_tr = model_logreg.predict(x_train)
stop_time=time.time()
total_time=stop_time-start_time
exec_time.append(total_time)
test_accuracy.append(sum(y_test == y_pred)/len(y_test))
train_accuracy.append(sum(y_train == y_pred_tr)/len(y_train))
print('Logistic Regression')
print('Test accuracy', sum(y_test == y_pred)/len(y_test))
print('Train accuracy', sum(y_train == y_pred_tr)/len(y_train))

start_time=time.time()
model_rfc=RandomForestClassifier()
model_rfc.fit(x_train, y_train)
y_pred = model_rfc.predict(x_test)
y_pred_tr = model_rfc.predict(x_train)
stop_time=time.time()
total_time=stop_time-start_time
exec_time.append(total_time)
test_accuracy.append(sum(y_test == y_pred)/len(y_test))
train_accuracy.append(sum(y_train == y_pred_tr)/len(y_train))
print('Random Forest Classifier')
print('Test accuracy', sum(y_test == y_pred)/len(y_test))
print('Train accuracy', sum(y_train == y_pred_tr)/len(y_train))

start_time=time.time()
model_mnb=MultinomialNB()
model_mnb.fit(x_train, y_train)
y_pred = model_mnb.predict(x_test)
y_pred_tr = model_mnb.predict(x_train)
stop_time=time.time()
total_time=stop_time-start_time
exec_time.append(total_time)
test_accuracy.append(sum(y_test == y_pred)/len(y_test))
train_accuracy.append(sum(y_train == y_pred_tr)/len(y_train))
print('Naive Bayes')
print('Test accuracy', sum(y_test == y_pred)/len(y_test))
print('Train accuracy', sum(y_train == y_pred_tr)/len(y_train))

start_time=time.time()
model_knn=KNeighborsClassifier(n_neighbors=5)
model_knn.fit(x_train, y_train)
y_pred = model_knn.predict(x_test)
y_pred_tr = model_knn.predict(x_train)
stop_time=time.time()
total_time=stop_time-start_time
exec_time.append(total_time)
test_accuracy.append(sum(y_test == y_pred)/len(y_test))
train_accuracy.append(sum(y_train == y_pred_tr)/len(y_train))
print('KNN')
print('Test accuracy', sum(y_test == y_pred)/len(y_test))
print('Train accuracy', sum(y_train == y_pred_tr)/len(y_train))

start_time=time.time()
model_dt=tree.DecisionTreeClassifier()
model_dt.fit(x_train, y_train)
y_pred = model_dt.predict(x_test)
y_pred_tr = model_dt.predict(x_train)
stop_time=time.time()
total_time=stop_time-start_time
exec_time.append(total_time)
test_accuracy.append(sum(y_test == y_pred)/len(y_test))
train_accuracy.append(sum(y_train == y_pred_tr)/len(y_train))
print('Decision Tree Classifier')
print('Test accuracy', sum(y_test == y_pred)/len(y_test))
print('Train accuracy', sum(y_train == y_pred_tr)/len(y_train))

start_time=time.time()
model_svm=SVC()
model_svm.fit(x_train, y_train)
y_pred = model_svm.predict(x_test)
y_pred_tr = model_svm.predict(x_train)
stop_time=time.time()
total_time=stop_time-start_time
exec_time.append(total_time)
test_accuracy.append(sum(y_test == y_pred)/len(y_test))
train_accuracy.append(sum(y_train == y_pred_tr)/len(y_train))
print('SVM')
print('Test accuracy', sum(y_test == y_pred)/len(y_test))
print('Train accuracy', sum(y_train == y_pred_tr)/len(y_train))
Output :
Logistic Regression
Test accuracy 0.9375752709755119
Train accuracy 0.9581743960382788
Random Forest Classifier
Test accuracy 0.9779205138498595
Train accuracy 0.9997992371009837
Naive Bayes
Test accuracy 0.8723404255319149
Train accuracy 0.8855651475607308
KNN
Test accuracy 0.9104777197912485
Train accuracy 0.9380311851703138
Decision Tree Classifier
Test accuracy 0.9684865515857085
Train accuracy 0.9997992371009837
SVM
Test accuracy 0.9668807707747893
Train accuracy 0.9933748243324634
from sklearn.ensemble import BaggingClassifier
start_time=time.time()
model_bc=BaggingClassifier()
model_bc.fit(x_train, y_train)
y_pred = model_bc.predict(x_test)
y_pred_tr = model_bc.predict(x_train)
stop_time=time.time()
total_time=stop_time-start_time
exec_time.append(total_time)
test_accuracy.append(sum(y_test == y_pred)/len(y_test))
train_accuracy.append(sum(y_train == y_pred_tr)/len(y_train))
print('Bagging Classifier')
print('Test accuracy', sum(y_test == y_pred)/len(y_test))
print('Train accuracy', sum(y_train == y_pred_tr)/len(y_train))
Output :
Bagging Classifier
Test accuracy 0.9690887193898033
Train accuracy 0.9971893194137723
from sklearn.ensemble import GradientBoostingClassifier
start_time=time.time()
model_gbc=GradientBoostingClassifier()
model_gbc.fit(x_train, y_train)
y_pred = model_gbc.predict(x_test)
y_pred_tr = model_gbc.predict(x_train)
stop_time=time.time()
total_time=stop_time-start_time
exec_time.append(total_time)
test_accuracy.append(sum(y_test == y_pred)/len(y_test))
train_accuracy.append(sum(y_train == y_pred_tr)/len(y_train))
print('Gradient Boosting Classifier')
print('Test accuracy', sum(y_test == y_pred)/len(y_test))
print('Train accuracy', sum(y_train == y_pred_tr)/len(y_train))

Output :
Gradient Boosting Classifier
Test accuracy 0.9056603773584906
Train accuracy 0.9099913002743759
from sklearn.linear_model import RidgeClassifier
start_time=time.time()
model_rc=RidgeClassifier()
model_rc.fit(x_train, y_train)
y_pred = model_rc.predict(x_test)
y_pred_tr = model_rc.predict(x_train)
stop_time=time.time()
total_time=stop_time-start_time
exec_time.append(total_time)
test_accuracy.append(sum(y_test == y_pred)/len(y_test))
train_accuracy.append(sum(y_train == y_pred_tr)/len(y_train))
print('Ridge Classifier')
print('Test accuracy', sum(y_test == y_pred)/len(y_test))
print('Train accuracy', sum(y_train == y_pred_tr)/len(y_train))
Output:
Ridge Classifier
Test accuracy 0.960457647531112
Train accuracy 0.9862142809342167

5. Comparison of Accuracies and Execution Times
Bar graphs are plotted using both the test accuracies and execution times of every classification algorithm.

models=['LogReg','RandomFC','NaiveBayes','KNN','DecisionTree','SVM','BC','GBC','RC']
fig = plt.figure()
plt.figure(figsize=(10,5))
sns.barplot(x=models,y=test_accuracy)
plt.ylim([0.8,1.0])
plt.show()


fig = plt.figure()
plt.figure(figsize=(10,5))
sns.barplot(x=models,y=exec_time)
plt.title('Plot between Execution Times and Algorithm')
plt.show()

Observations :
It is observed that the test accuracies performed in the order of Random Forest Classifier, Decision Tree Classifier, SVM, Logistic Regression, KNN and Naive Bayes.

In terms of execution time Naive Bayes Algorithm is the fastest where as Gradient Boosting Classifier, SVM are slower.

Classification Report for Random Forest Classifier
Classification Report for Random Forest Classifier is generated.

As it is observed that, test F1 scores for all the three classes are good enough, Random Forest Classifier can be used as best algorithm for reviews classification.


Tuning Hyperparameters
Random Forest Classifier algorithm's hyperparameters of n_estimators and criterion are tuned in order to check for the maximum test accuracy.

For both the 'gini index' and 'entropy' as criterion, for n_estimators values of 10,50,100,120,150,200,400, test accuracies are recorded.

Bar graphs are plotted for the recorded test accuracies.

def rfc_gini(nestimators):
 model_rfc=RandomForestClassifier(n_estimators=nestimators,criterion="gini")
 model_rfc.fit(x_train, y_train)
 y_pred = model_rfc.predict(x_test)
 y_pred_tr = model_rfc.predict(x_train)
 print('Random Forest Classifier')
 print('For entropy criterion and n_estimators = ',nestimators)
 print('Test accuracy', sum(y_test == y_pred)/len(y_test))
 exp1.append(sum(y_test == y_pred)/len(y_test))
gini_nest=[10,50,100,120,150,200,400]
exp1=[]
for i in gini_nest:
    rfc_gini(i)
Random Forest Classifier
For entropy criterion and n_estimators =  10
Test accuracy 0.9731031714171016
Random Forest Classifier
For entropy criterion and n_estimators =  50
Test accuracy 0.9777197912484946
Random Forest Classifier
For entropy criterion and n_estimators =  100
Test accuracy 0.9781212364512244
Random Forest Classifier
For entropy criterion and n_estimators =  120
Test accuracy 0.9781212364512244
Random Forest Classifier
For entropy criterion and n_estimators =  150
Test accuracy 0.9781212364512244
Random Forest Classifier
For entropy criterion and n_estimators =  200
Test accuracy 0.9783219590525893
Random Forest Classifier
For entropy criterion and n_estimators =  400
Test accuracy 0.9785226816539543


def rfc_entropy(nestimators):
 model_rfc=RandomForestClassifier(n_estimators=nestimators,criterion="entropy")
 model_rfc.fit(x_train, y_train)
 y_pred = model_rfc.predict(x_test)
 y_pred_tr = model_rfc.predict(x_train)
 print('Random Forest Classifier')
 print('For entropy criterion and n_estimators = ',nestimators)
 print('Test accuracy', sum(y_test == y_pred)/len(y_test))
 exp2.append(sum(y_test == y_pred)/len(y_test))
nest=[10,50,100,120, 150,200,400]
exp2=[]
for i in nest:
    rfc_entropy(i)
def rfc_entropy(nestimators):
 model_rfc=RandomForestClassifier(n_estimators=nestimators,criterion="entropy")
 model_rfc.fit(x_train, y_train)
 y_pred = model_rfc.predict(x_test)
 y_pred_tr = model_rfc.predict(x_train)
 print('Random Forest Classifier')
 print('For entropy criterion and n_estimators = ',nestimators)
 print('Test accuracy', sum(y_test == y_pred)/len(y_test))
 exp2.append(sum(y_test == y_pred)/len(y_test))
nest=[10,50,100,120, 150,200,400]
exp2=[]
for i in nest:
    rfc_entropy(i)
Random Forest Classifier
For entropy criterion and n_estimators =  10
Test accuracy 0.9704937775993577
Random Forest Classifier
For entropy criterion and n_estimators =  50
Test accuracy 0.9779205138498595
Random Forest Classifier
For entropy criterion and n_estimators =  100
Test accuracy 0.9779205138498595
Random Forest Classifier
For entropy criterion and n_estimators =  120
Test accuracy 0.9775190686471297
Random Forest Classifier
For entropy criterion and n_estimators =  150
Test accuracy 0.9775190686471297
Random Forest Classifier
For entropy criterion and n_estimators =  200
Test accuracy 0.9779205138498595
Random Forest Classifier
For entropy criterion and n_estimators =  400
Test accuracy 0.9775190686471297
fig = plt.figure()
plt.figure(figsize=(10,5))
sns.barplot(x=nest,y=exp2)
plt.ylim([0.97,0.98])
plt.title('Plot of n_estimators vs Test Accuracy for Entropy')
plt.show()

As the Naive Bayes Algorithm is the fastest, I've tried to tune the hyper parameters of NBC to increase the test accuracy.

Upon experimenting with different alpha values, for alpha = 0.1, test accuracy of NBC increased from 87 to 95.

def multinomialNB(alpha):
 model_mnb=MultinomialNB(alpha=alpha)
 model_mnb.fit(x_train, y_train)
 y_pred = model_mnb.predict(x_test)
 test_accuracy.append(sum(y_test == y_pred)/len(y_test))
 print('Naive Bayes')
 print('Alpha value is ',alpha)
 print('Test accuracy', sum(y_test == y_pred)/len(y_test))
 exp3.append(sum(y_test == y_pred)/len(y_test))
alpha=[0.1,0.2,0.3,0.5,1.0]
exp3=[]
for i in alpha:
    multinomialNB(i)
Naive Bayes
Alpha value is  0.1
Test accuracy 0.9500200722601365
Naive Bayes
Alpha value is  0.2
Test accuracy 0.9383781613809715
Naive Bayes
Alpha value is  0.3
Test accuracy 0.9289441991168206
Naive Bayes
Alpha value is  0.5
Test accuracy 0.9096748293857888
Naive Bayes
Alpha value is  1.0
Test accuracy 0.8723404255319149
fig = plt.figure()
plt.figure(figsize=(10,5))
sns.barplot(x=alpha,y=exp3)
plt.ylim([0.85,0.96])
plt.title('Plot of Alpha vs Test Accuracy for Naive Bayes Classifier')
plt.show()

Experiments and Contribution :
Through the exploratory analysis of the existing features of the dataset, I've followed the consideration of features and pre-processing of the data.

          Initially, reviews and ratings are considered. Reviews is concatenated with the Summary or Title. Data of booth the csv files provided in dataset is combined and used. Helpfulness isn't considered as the total votes aren't available in Large Dataset file of amazon and flipkart reviews.

I've spent time to understand about the Word Cloud, Stop Words, NLTK, Tokenization.

          Initially performed the basic cleaning of data followed by the application of Stop Words removal and tokenization. Utilized the Word Cloud for the data visualization. Implemented the mono gram analysis to visualize the frequent words in all classes.

I've spent time on understanding the classification algorithms available with the sklearn libraries.

          Applied the classifiers from the libraries to build different classification models.

Tracking the performance of all the classification algorithms, compared them by plotting the test accuracies and execution times.

           In terms of test accuracy, classification algorithms performance was in the order of Random Forest Classifier, Bagging Classifier, Decision Tree Classifier, Support Vector Machine, Ridge Classifier, Logistic Regression, KNN, Stochastic Gradient Boosting Classifier and Naive Bayes Classifier.

           In terms of execution time, Naive Bayes is the fastest and Stochastic Gradient Boosting Classifier and SVM are slower.

In the best performed Random Forest Classifier, test F1 scores for three classes are observed to be 0.96, 0.90, 0.99. This represents that the model is performing well and data is not partial towards any class.

Training accuracies of Random Forest Classifier, decision tree and SVM algorithms are very high and appear to be overfitting but since test accuracies are also performing well, this can't be considered as the overfitting.

Experiments of different n_estimators using both 'gini index' and 'entropy' as criterion are performed on Random Forest Classifier Model(as it is highest performing on this dataset) and highest test accuracy is observed when n_estimators = 400 and criterion = gini i.e 97.85.

Since Naive Bayes Classifier is the fastest, I performed experiments on increasing it's test accuracy also and increased the accuracy of NBC from 87 to 95 when alpha value is 0.1.

Challenges Faced :
Initially I faced a challenge in understanding the large data set and didn't understand how to consider it. 

          Later, started checking the info of all the features existing from all the files and confined to only few important features and thus solved the issue.

Didn't know the ways to tokenize, remove stop words but spent time to understand the python libraries and got the ways to perform pre processing of data.

Previously, I had no idea of all sklearn libraries of ML algorithms. So, spent some time on understanding the libraries, building models and understanding the hyper parameters of algorithmic functions.

References :
[1] https://www.kaggle.com/code/benroshan/sentiment-analysis-amazon-reviews 

[2] https://www.kaggle.com/code/lele1995/amazon-reviews-sentiment-analysis/ 

[3] https://towardsdatascience.com/pros-and-cons-of-various-classification-ml-algorithms-3b5bfb3c87d6 

[4] https://towardsdatascience.com/pros-and-cons-of-various-classification-ml-algorithms-3b5bfb3c87d6 

[5] https://machinelearningmastery.com/hyperparameters-for-classification-machine-learning-algorithms/ 

