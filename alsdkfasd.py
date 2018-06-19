import pandas as pd
import numpy as np
import nltk.data
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import naive_bayes
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from nltk.tokenize import sent_tokenize
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

ps = PorterStemmer()
def stem_tokens(text):
  words = word_tokenize(text)

  st = ""
  for w in words:
    st = st + " " + ps.stem(w)

  return ps.stem(text)


placemarker = "amphisbaena"
my_stopwords = set(stopwords.words('english')) - {'don', 't', 'against', 'no', 'not'}

def extract_clause(sentence):
    #chinking
    grammar = ('''
        NP: 
           {<.*>+} 
           }<\(|,>+{ # NP
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


def first():
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    df=pd.read_csv('data 2_train.csv', skiprows=1, names=['example_id', 'text', 'aspect_term', 'term_location', 'clasification'])
    #df

    df.text = df.text.str.replace('\[comma\]', ',')
    # print(df.text[94])
    # exit(0)
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
    # exit(0)


    stopset = set(stopwords.words('english')) - {'don', 't', 'against', 'no', 'not'}
    #TFIDF vectorize
    vectorizer=TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=stopset, preprocessor=stem_tokens, analyzer = 'word', ngram_range=(1, 3) )

    #dependent variable
    y=df.clasification

    #convert df.txt from text to feature
    X=vectorizer.fit_transform(text)

    print(y.shape)#observations
    print(X.shape)#unique words

    kf = KFold(n_splits=10, shuffle=True)
    a = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        svm_clf1 = svm.SVC(C=1, gamma=1, decision_function_shape='ovo')
        svm_clf1.fit(X_train, y_train)
        # svm_clf.score(X, y)
        svmpredict1 = svm_clf1.predict(X_test)
        print('svm  Test accuracy for dataset1 is ', metrics.accuracy_score(y_test, svmpredict1))
        print(metrics.classification_report(y_test, svmpredict1))
        a.append(metrics.accuracy_score(y_test, svmpredict1))




        # we will train a naive bayes classifier
        # clf = naive_bayes.MultinomialNB()
        # clf.fit(X_train, y_train)
        #
        # a.append(clf.score(X_test, y_test))

        # for i in test_index:
        #     c = clf.predict(X[i])
        #     if c != df.clasification[i]:
        #         print(text[i], "Predicted as", c, "but should be", df.clasification[i])

    print(a)
    print(np.average(a))


first()