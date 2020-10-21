#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import testsets
import evaluation
import loadData
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding, LSTM, Dense, Activation, SpatialDropout1D
from keras.models import Sequential
from sklearn.model_selection import GridSearchCV

#constant values
MAX_NUM_WORDS = 5000
MAX_LENGTH_SEQUENCE = 20
EMBEDDING_DIM = 100
BATCH_SIZE = 32

# TODO: load training data
trainingData = loadData.loadPreProcessedData("semeval-tweets/twitter-training-data.txt")

trainingData_idList = []
trainingData_sentimentList = []
trainingData_tweetList = []

for data in trainingData:
    trainingData_idList.append(data[0])
    trainingData_sentimentList.append(data[1])
    trainingData_tweetList.append(data[2])
#change the sentiment to numbers 
trainingData_sentimentList_num = []
for senti in trainingData_sentimentList:
    if senti == 'positive':
        trainingData_sentimentList_num.append(1)
    elif senti == 'negative':
        trainingData_sentimentList_num.append(0)
    elif senti == 'neutral':
        trainingData_sentimentList_num.append(2)
trainingData_sentimentList = trainingData_sentimentList_num

sentiment_index = {'positive':1, 'nagative':0, 'neutral':2}

# print(len(trainingData_idList))
# print(trainingData_idList[111])
# print(len(trainingData_sentimentList))
# print(trainingData_sentimentList[111])
# print(len(trainingData_tweetList))
# print(trainingData_tweetList[111])

count_vect = CountVectorizer()

tfidf_transformer = TfidfTransformer()
# ['NaiveBayesClassifier', 'SVMclassifier', 'LSTMclassifier']
for classifier in [ 'NaiveBayesClassifier', 'SVMclassifier', 'LSTMclassifier']: # You may rename the names of the classifiers to something more descriptive
    if classifier == 'NaiveBayesClassifier':
        print('Training ' + classifier)
        # TODO: extract features for training classifier1
        trainingDataTweetList_counts = count_vect.fit_transform(trainingData_tweetList)
        
        trainingDataTweetList_tfidf = tfidf_transformer.fit_transform(trainingDataTweetList_counts)

        # TODO: train sentiment classifier1
        clf_naiveBayes = MultinomialNB().fit(trainingDataTweetList_tfidf, trainingData_sentimentList)


    elif classifier == 'SVMclassifier':
        print('Training ' + classifier)
        # TODO: extract features for training classifier2
        clf_SVM = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                    alpha=1e-3, random_state=42,
                                    max_iter=5, tol=None)),
        ])

        parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
        'tfidf__use_idf': (True, False),
        'clf__alpha': (1e-2, 1e-3),}

        gs_clf_SVM = GridSearchCV(clf_SVM, parameters, cv=5, n_jobs=-1)

        # TODO: train sentiment classifier2
        gs_clf_SVM.fit(trainingData_tweetList, trainingData_sentimentList)


    elif classifier == 'LSTMclassifier':
        print('Training ' + classifier)
        # TODO: extract features for training classifier3
        #load glove data
        gloveData_index = loadData.loadGloveData_index("glove/glove.6B.100d.txt")
        # print(gloveData_index['the'])
        # print(len(gloveData_index))
        #tokenize
        tokenizer = Tokenizer(num_words = MAX_NUM_WORDS)
        tokenizer.fit_on_texts(trainingData_tweetList)
        #get sequences index of training data
        training_tokenizer_sequences = tokenizer.texts_to_sequences(trainingData_tweetList)
        #get unique index for every word
        training_tokenizer_word_index = tokenizer.word_index
        # print('----------------------', len(training_tokenizer_word_index))
        # print(training_tokenizer_word_index['norris'],training_tokenizer_word_index['taxi'],
        #     training_tokenizer_word_index['song'],training_tokenizer_word_index['stuck'],
        #     training_tokenizer_word_index['head'])
        # print(trainingData_tweetList[111])
        # print(training_tokenizer_sequences[111])
        #take same length(the maximum)
        trainingDataTweetList_inSequence = pad_sequences(training_tokenizer_sequences, 
            maxlen = MAX_LENGTH_SEQUENCE)
        trainingDataSentimentList_inArray = to_categorical(np.asarray(trainingData_sentimentList))
        # print(len(trainingDataTweetList_inSequence),len(trainingDataSentimentList_inArray))
        #generate Embedding Matrix
        maxNum_words = min(MAX_NUM_WORDS, len(training_tokenizer_word_index))
        EmbeddingMatrix = np.zeros((maxNum_words + 1, EMBEDDING_DIM))
        for w, i in training_tokenizer_word_index.items():
            if i <= MAX_NUM_WORDS:
                EmbeddingMatrix_vector = gloveData_index.get(w)
                if EmbeddingMatrix_vector is not None:
                    EmbeddingMatrix[i] = EmbeddingMatrix_vector
        # print(EmbeddingMatrix.shape)
        # print(gloveData_index['boss'])
        # print(EmbeddingMatrix[1843])


        # TODO: train sentiment classifier3
        print("building the NEURAL MODEL...")
        embeddingLayer = Embedding(maxNum_words+1, EMBEDDING_DIM, 
            weights=[EmbeddingMatrix], input_length=MAX_LENGTH_SEQUENCE, trainable = False)
        model = Sequential()
        model.add(embeddingLayer)
        model.add(SpatialDropout1D(0.4))
        model.add(LSTM(100, dropout =0.2, recurrent_dropout=0.2))
        # model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.add(Dense(3, activation = 'softmax'))
        # model.layers[0].trainable=False
        # print(model.layers[0].name,model.layers[0].trainable,"=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', 
            metrics=['accuracy'])
        print("training the NEURAL MODEL...")
        print(model.summary())
        model.fit(trainingDataTweetList_inSequence, 
            trainingDataSentimentList_inArray, batch_size=BATCH_SIZE, epochs=5, verbose = 2)









    for testset in testsets.testsets:
        # TODO: classify tweets in test set
        #load test data
        testData = loadData.loadPreProcessedData(testset)

        testData_idList = []
        testData_sentimentList = []
        testData_tweetList = []

        for data in testData:
            testData_idList.append(data[0])
            testData_sentimentList.append(data[1])
            testData_tweetList.append(data[2])

        testData_sentimentList_num = []
        for senti in testData_sentimentList:
            if senti == 'positive':
                testData_sentimentList_num.append(1)
            elif senti == 'negative':
                testData_sentimentList_num.append(0)
            elif senti == 'neutral':
                testData_sentimentList_num.append(2)

        testData_sentimentList = testData_sentimentList_num


        # print(testData_tweetList[5])

        

        #perform predictions basing on different classifiers
        if classifier == 'NaiveBayesClassifier':
            testDataTweetList_counts = count_vect.transform(testData_tweetList)
            testDataTweetList_tfidf = tfidf_transformer.transform(testDataTweetList_counts)
            predictedSentiment = clf_naiveBayes.predict(testDataTweetList_tfidf)
            # print(predictedSentiment[:100])
        elif classifier == 'SVMclassifier':
            predictedSentiment = gs_clf_SVM.predict(testData_tweetList)
            print(gs_clf_SVM.best_score_)
            for param_name in sorted(parameters.keys()):
                print("%s: %r" % (param_name, gs_clf_SVM.best_params_[param_name]))
            # print(predictedSentiment[:100])
        elif classifier == 'LSTMclassifier':
            test_tokenizer_sequences = tokenizer.texts_to_sequences(testData_tweetList)
            testDataTweetList_inSequence = pad_sequences(test_tokenizer_sequences, 
            maxlen = MAX_LENGTH_SEQUENCE)

            predictedSentiment = model.predict_classes(testDataTweetList_inSequence)
            # print(len(predictedSentiment))
            # print(predictedSentiment[:100])

        predictions = {}
        for tweetid, sentiment in zip(testData_idList, predictedSentiment):
            if sentiment == 1:
                predictions[tweetid] = 'positive'
            elif sentiment == 0:
                predictions[tweetid] = 'negative'
            elif sentiment == 2:
                predictions[tweetid] = 'neutral'
        # print(predictions)
        # print(len(predictedSentiment), len(testData_sentimentList))
        # print(np.mean(predictedSentiment == testData_sentimentList))








        # predictions = {'163361196206957578': 'neutral', '768006053969268950': 'neutral', '742616104384772304': 'neutral', '102313285628711403': 'neutral', '653274888624828198': 'neutral'} # TODO: Remove this line, 'predictions' should be populated with the outputs of your classifier
        evaluation.evaluate(predictions, testset, classifier)

        evaluation.confusion(predictions, testset, classifier)
