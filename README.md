## This repository contains some of my experimental *Machine Learning* projects:

# 1) Discrimination of benign and malignant mammographic masses: 
In this project, we'll be using the "mammographic masses" public dataset from the UCI repository (source: https://archive.ics.uci.edu/ml/datasets/Mammographic+Mass)

This data contains 961 instances of masses detected in mammograms, and contains the following attributes:


   1. BI-RADS assessment: 1 to 5 (ordinal)  
   2. Age: patient's age in years (integer)
   3. Shape: mass shape: round=1 oval=2 lobular=3 irregular=4 (nominal)
   4. Margin: mass margin: circumscribed=1 microlobulated=2 obscured=3 ill-defined=4 spiculated=5 (nominal)
   5. Density: mass density high=1 iso=2 low=3 fat-containing=4 (ordinal)
   6. Severity: benign=0 or malignant=1 (binominal)
   
BI-RADS is an assesment of how confident the severity classification is; it is not a "predictive" attribute and so we will discard it. The age, shape, margin, and density attributes are the features that we will build our model with, and "severity" is the classification we will attempt to predict based on those attributes.

Although "shape" and "margin" are nominal data types, which sklearn typically doesn't deal with well, they are close enough to ordinal that we shouldn't just discard them. The "shape" for example is ordered increasingly from round to irregular.

A lot of unnecessary anguish and surgery arises from false positives arising from mammogram results. If we can build a better way to interpret them through supervised machine learning, it could improve a lot of lives.

*The Project has following two parts:*
## (a) Mammographic_Masses_MLP:
In part (a) of the project, I applied *Multi Layer Perceptron (MLP)* to classify benign and malignant mamographic masses given certain medical attributes.

## (b) Mammographic_Masses_SML:
In this part (b) of the project I used several different *supervised machine learning (SML)* techniques to this data set, and see which one yields the highest accuracy as measured with K-Fold cross validation (K=19). Apply:

* Decision tree
* Random forest
* KNN
* Naive Bayes
* SVM
* Logistic Regression

**Project Courtesy**: Mr. Frank Kane (http://sundog-education.com)

# 2) IMDB_keras_RNN:

This notebook is inspired by the imdb_lstm.py example that ships with Keras. It's actually a great example of using Recurrent Neural Network's (RNN). The data set we're using consists of user-generated movie reviews and classification of whether the user liked the movie or not based on its associated rating.

More info on the dataset is here:
https://keras.io/datasets/#imdb-movie-reviews-sentiment-classification

So we are going to use an RNN to do sentiment analysis on full-text movie reviews!
Think about how amazing this is. We're going to train an artificial neural network how to "read" movie reviews and guess whether the author liked the movie or not from them.

Since understanding written language requires keeping track of all the words in a sentence, we need a recurrent neural network to keep a "memory" of the words that have come before as it "reads" sentences over time.

In particular, we'll use LSTM (Long Short-Term Memory) cells because we don't really want to "forget" words too quickly - words early on in a sentence can affect the meaning of that sentence significantly.

# 3) StudentsAdmision@UCLA_MLP: 
In this project, we predict student admissions to graduate school at UCLA based on three pieces of data:
- GRE Scores (Test)
- GPA Scores (Grades)
- Class rank (1-4)

The dataset originally came from here: http://www.ats.ucla.edu/


# 4) IMDB_keras_MLP:

This project involves a dataset of 25,000 IMDB reviews. Each review, comes with a label. A label of 0 is given to a negative review, and a label of 1 is given to a positive review. The goal of this lab is to create a model that will predict the sentiment of a review, based on the words on it. You can see more information about this dataset in the Keras website.

Now, the input already comes preprocessed for us for convenience. Each review is encoded as a sequence of indexes, corresponding to the words in the review. The words are ordered by frequency, so the integer 1 corresponds to the most frequent word ("the"), the integer 2 to the second most frequent word, etc. By convention, the integer 0 corresponds to unknown words.

Then, the sentence is turned into a vector by simply concatenating these integers. For instance, if the sentence is "To be or not to be." and the indices of the words are as follows:

- "to": 5
- "be": 8
- "or": 21
- "not": 3

Then the sentence gets encoded as the vector [5,8,21,3,5,8].
