import nltk

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
            clause = clause + i[0] + " "

        # clause = clause + "\n"
        if clause.find('amphisbaena') > -1:
            return clause

# tree.draw()

sentence = "They did not have mayonnaise, forgot our toast left out ingredients (ie amphisbaena in an omelet), below hot temperatures and the bacon was so over cooked it crumbled on the plate when you touched it."
extract_clause(sentence)
