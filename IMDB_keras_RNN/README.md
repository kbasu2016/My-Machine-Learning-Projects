This notebook is inspired by the imdb_lstm.py example that ships with Keras. It's actually a great example of using RNN's. 
The data set we're using consists of user-generated movie reviews and classification of whether the user liked the movie or not 
based on its associated rating.

More info on the dataset is here:
https://keras.io/datasets/#imdb-movie-reviews-sentiment-classification

So we are going to use an RNN to do sentiment analysis on full-text movie reviews!

Think about how amazing this is. We're going to train an artificial neural network how to "read" movie reviews and guess whether the 
author liked the movie or not from them. Since understanding written language requires keeping track of all the words in a sentence, 
we need a recurrent neural network to keep a "memory" of the words that have come before as it "reads" sentences over time.
In particular, we'll use LSTM (Long Short-Term Memory) cells because we don't really want to "forget" words too quickly - words 
early on in a sentence can affect the meaning of that sentence significantly.

Project Idea: Mr. Frank Kane.
