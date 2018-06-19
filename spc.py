import pandas as pd
import spacy
from spacy import displacy
# from spacy.en import English

nlp = spacy.load('en_core_web_sm')


df = pd.read_csv('data 2_train.csv', skiprows=1,
                 names=['example_id', 'text', 'aspect_term', 'term_location', 'clasification'])

df.text = df.text.str.replace('\[comma\]', ',')

doc = nlp(df.text[16])
# displacy.serve(doc, style='dep')


# nlp = English()
# doc = nlp(df.text[16])
for sent in doc:
    print(sent)
    # for token in sent:
    #     if token.is_alpha:
    #         print(token.orth_, token.tag_, token.head.lemma_)