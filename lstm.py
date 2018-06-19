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

df.text = df.text.str.replace('\[comma\]', ',')

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

# stopset = set(stopwords.words('english')) - {'don', 't', 'against', 'no', 'not'}
# TFIDF vectorize
vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii')


# dependent variable
y = df.clasification

le = LabelEncoder()
Y = le.fit_transform(y)
Y = Y.reshape(-1,1)

X_train,X_test,Y_train,Y_test = train_test_split(text,Y,test_size=0.15, shuffle=True)


# print("labels", Y_train)

max_words = 2000
max_len = 200
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('tanh')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model

model = RNN()
model.summary()
model.compile(loss='hinge',optimizer=RMSprop(),metrics=['accuracy'])

model.fit(sequences_matrix,Y_train,batch_size=32,epochs=10,
          validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])


test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)


accr = model.evaluate(test_sequences_matrix,Y_test)

print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
