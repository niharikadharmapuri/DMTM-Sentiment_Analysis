import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('sentiwordnet')
nltk.download('wordnet')

import pandas as pd
from nltk.corpus import sentiwordnet as swn #importing the sentiwordnet
from nltk.tag.perceptron import PerceptronTagger #importing a tagger
from nltk.tokenize import word_tokenize #of course tokenizing the input
tagger=PerceptronTagger() #init the tagger in default mode


df1=pd.read_csv('data 1_train.csv', skiprows=1, names=['example_id', 'text', 'aspect_term', 'term_location', 'clas'])
#df1
aspect_list=[]
aspect_list=df1.aspect_term# storing the aspect terms in a list

for a in aspect_list:
    for aspect, tag in tagger.tag(word_tokenize('delete key')):
        print(a)
        print(aspect)
        print(tag)
        if tag=="JJ" or tag=='JJR' or tag == 'JJS' and swn.senti_synsets(aspect, "a") == True: #check if the word is an adjective
            print("x", aspect, "y", list(swn.senti_synsets(aspect, "a")))
            synset=list(swn.senti_synsets(aspect, "a"))[0] #get the most likely synset
            print(synset) #print out the values