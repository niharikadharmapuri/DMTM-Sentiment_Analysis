# LSTM for sequence classification in the IMDB dataset
import nltk
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Convolution1D, Flatten, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import TensorBoard
from keras.preprocessing.text import Tokenizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping


placemarker = "amphisbaena"
my_stopwords = set(stopwords.words('english')) - {'don', 't', 'against', 'no', 'not'}

def extract_clause(sentence):
    #chinking
    grammar = ('''
        NP: 
           {<.*>+} 
           }<\(|,|CC>+{ # NP
        ''')


    # text=nltk.word_tokenize(sentence)
    # print(nltk.pos_tag(text))

    chunkParser = nltk.RegexpParser(grammar)
    tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    tree = chunkParser.parse(tagged)
    for subtree in tree.subtrees(lambda t: t.height() == 2):
        # if 'bacon' in subtree:
        # if subtree.pos() > 0:
        clause = ""
        for i in subtree.leaves():
            if i[0] not in my_stopwords:
                clause = clause + i[0] + " "

        # clause = clause + "\n"
        if clause.find(placemarker) > -1:
            return clause


df = pd.read_csv('data 1_train.csv', skiprows=1,
                 names=['example_id', 'text', 'aspect_term', 'term_location', 'clasification'])
# df

text = []
for i in range(len(df)):
    # df.text = df.text.str.replace(df.aspect_term, 'ASPECT')
    s = int(df.term_location[i].split('--')[0])
    t = int(df.term_location[i].split('--')[1])

    interesting_clause = extract_clause(df.text[i][0:s] + placemarker + df.text[i][t:])
    # print(i, interesting_clause, df.clasification[i])
    text.append(interesting_clause)
    # print(df.text[i][x:y])
    # print(sent_detector.tokenize(df.text[i]))
    # print(sent_tokenize(df.text[i]))

# print(text)
# print(type(df.text))
text = pd.Series(text).astype(str)

df.text = df.text.str.replace('\[comma\]', ',')
print(df.text[2])

stopset = set(stopwords.words('english')) - {'don', 't', 'against', 'no', 'not'}
# TFIDF vectorize
vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=stopset)


# dependent variable
y = df.clasification










# convert df.txt from text to feature
# X = vectorizer.fit_transform(df.text)
max_review_length = 3200
t = Tokenizer(num_words=max_review_length)

t.fit_on_texts(text)
# summarize what was learned
print(t.word_counts)
print(t.document_count)
print(t.word_index)
print(t.word_docs)

X = t.texts_to_matrix(text, mode='count')

print("0", X[0])

print(y.shape)  # observations
print(X.shape)  # unique words

kf = KFold(n_splits=10, shuffle=True)
a = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Using keras to load the dataset with the top_words
    top_words = 10000
    # (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

    # Pad the sequence to the same length
    X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

    # Using embedding from Keras
    embedding_vector_length = 300
    model = Sequential()
    model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))

    # Convolutional model (3x conv, flatten, 2x dense)
    model.add(Convolution1D(64, 3, padding='same'))
    model.add(Convolution1D(32, 3, padding='same'))
    model.add(Convolution1D(16, 3, padding='same'))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(180,activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(1,activation='sigmoid'))

    # Log to tensorboard
    tensorBoardCallback = TensorBoard(log_dir='./logs', write_graph=True)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=3, callbacks=[tensorBoardCallback], batch_size=64)

    # Evaluation on the test set
    scores = model.evaluate(X_test, y_test, verbose=1)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    print(model.metrics_names)
    print(scores)
    exit(0)