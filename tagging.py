 #Copywrite Warning: Owner of the code is Gulcheera Academy(Khosiyat Sabirova)
                                                        #This code can be used by anyone for free, but the name "Gulcheera Academy" must be acknowledged 
#Part of Speech Tagging with NLTK

#nltk packages are imported
import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

example_4Tagging1 = state_union.raw("2005-GWBush.txt")#create a variable to store a raw data which is in text format provided by the corpus of nltk package
example_4Tagging2 = state_union.raw("2006-GWBush.txt")

#define a function to tag the contents
def contentTagging(sample_text, train_text):
    tokenized_trained = PunktSentenceTokenizer(train_text)#create a variable to store a tokenized and trained body of the text (for unsupervised machine learning)
    tokenized = tokenized_trained.tokenize(sample_text)#create a variable to store tokenized sample text
    try:
        #loop through the tokanized sample text
        for lexUnit in tokenized[:10]:
            words = nltk.word_tokenize(lexUnit)#create a variable to store the tokanized words those have been looped through
            tagged = nltk.pos_tag(words)#tag the tokenized words and store them in a variable
            print(tagged)

    except Exception as skip:
        print(str(skip))

#print the result
contentTagging(example_4Tagging1,example_4Tagging2)
