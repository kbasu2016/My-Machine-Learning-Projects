## This repository contains some of my experimental ML projects:

# Mammographic_Masses: 
I applied *Multi Layer Perceptron (MLP)* to classify benign and malignant mamographic masses given certain medical attributes

# IMDB_keras_RNN:

This notebook is inspired by the imdb_lstm.py example that ships with Keras. It's actually a great example of using RNN's. The data set we're using consists of user-generated movie reviews and classification of whether the user liked the movie or not based on its associated rating.

More info on the dataset is here:
https://keras.io/datasets/#imdb-movie-reviews-sentiment-classification

So we are going to use an RNN to do sentiment analysis on full-text movie reviews!
Think about how amazing this is. We're going to train an artificial neural network how to "read" movie reviews and guess whether the author liked the movie or not from them.

Since understanding written language requires keeping track of all the words in a sentence, we need a recurrent neural network to keep a "memory" of the words that have come before as it "reads" sentences over time.

In particular, we'll use LSTM (Long Short-Term Memory) cells because we don't really want to "forget" words too quickly - words early on in a sentence can affect the meaning of that sentence significantly.

# StudentsAdmision@UCLA_MLP: 
In this project, we predict student admissions to graduate school at UCLA based on three pieces of data:
- GRE Scores (Test)
- GPA Scores (Grades)
- Class rank (1-4)

The dataset originally came from here: http://www.ats.ucla.edu/


# IMDB_keras_MLP:

This project involves a dataset of 25,000 IMDB reviews. Each review, comes with a label. A label of 0 is given to a negative review, and a label of 1 is given to a positive review. The goal of this lab is to create a model that will predict the sentiment of a review, based on the words on it. You can see more information about this dataset in the Keras website.

Now, the input already comes preprocessed for us for convenience. Each review is encoded as a sequence of indexes, corresponding to the words in the review. The words are ordered by frequency, so the integer 1 corresponds to the most frequent word ("the"), the integer 2 to the second most frequent word, etc. By convention, the integer 0 corresponds to unknown words.

Then, the sentence is turned into a vector by simply concatenating these integers. For instance, if the sentence is "To be or not to be." and the indices of the words are as follows:
- "to": 5
- "be": 8
- "or": 21
- "not": 3

Then the sentence gets encoded as the vector [5,8,21,3,5,8].
