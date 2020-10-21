import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np

def loadPreProcessedData(DataSet):
	id_gt_tweetsList = []
	with open(DataSet, 'r') as f:
		for line in f:
			line = line.lower()
			line = line.rstrip('\n')
			id_gt_tweetsList_single = line.split('\t')
			tweet = id_gt_tweetsList_single[2]

			#remove user mentions
			tweet = re.sub(r'@\S+\s*', ' ', tweet)
			#remove hashtags
			tweet = re.sub(r'#\S+\s*', ' ', tweet)
			#remove http links
			tweet = re.sub(r'http\S+\s*', ' ', tweet)
			#remove on-alphanumeric characters except spaces
			tweet = re.sub(r'[^A-Za-z0-9 ]+', ' ', tweet)
			#remove words with only 1 character
			tweet = re.sub(r'\b([a-zA-Z])\b', '', tweet)
			#reomve numbers that are fully made of digits
			tweet = re.sub(r'\b\d+\b *', '', tweet)
			#remove extra whitespace
			tweet = re.sub(r'\s+', ' ', tweet)

			#remove stop words
			stop_words = set(stopwords.words('english'))
			tweet_tokens = word_tokenize(tweet)
			filtered_tweet_tokens = [w for w in tweet_tokens if not w in stop_words]
			tweet = ' '.join(filtered_tweet_tokens)


			id_gt_tweetsList_single[2] = tweet
			id_gt_tweetsList.append(id_gt_tweetsList_single)
		# print(id_gt_tweetsList[6])

	f.close()
	return id_gt_tweetsList

def loadGloveData_index(gloveDataSet):
	index = {}
	with open(gloveDataSet,'r') as fi:
		for line in fi:
			tokens = line.split()
			words = tokens[0]
			vectors = np.asarray(tokens[1:], dtype = 'float32')
			index[words] = vectors
	fi.close()
	return index










