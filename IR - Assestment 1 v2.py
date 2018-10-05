
'''                 Created by 1700927 and 1700239                          '''
'''                  last time edited: 22-02-2018                        '''

## Libraries
import nltk
from nltk.corpus import stopwords
from urllib import request
from bs4 import BeautifulSoup
import string
import math
import collections
import operator
import pandas as pd
from operator import itemgetter

'''         Reading Web Pages        '''

# List of webpages
#url = ['http://csee.essex.ac.uk/staff/udo/index.html','http://irsg.bcs.org/ksjaward.php']

## Typing - The user must type, one by one, the URL that are going to be indexed
url = []
exit = '1'
print('Enter a valid URL')
print('You can type as many URLs as you want')
print('When you have finished, type "0" (zero)')
while (exit == '1'):
    aa = input('Type URL: ')
    if (aa != '0'):
        url.append(aa)
    else:
        exit = aa

# Print list of URL typed
print('URLs stored: ', url)
print('\n')
# Creates a list where after processing every url, which means after tokenization, NER, POS-Tag, and so on, all this final words
# will be stored.
ListWords = []

## Extracting information from a URL
for i in range(0, len(url)):

    ## Read and store all the content of the "i" URL
    web = request.urlopen(url[i])
    raw = BeautifulSoup(web.read().decode('utf8'), "lxml")
    print('Raw text : ', url)
    print(raw)
    print('\n')

    '''     Parsing HTML    '''

    ## Gets the information stored from different Meta tags
    metatags = raw.find_all('meta')
    metatext = [meta.attrs['content'] for meta in metatags]
    textweb = raw.get_text()
    print('MetaTags: ')
    print(metatext)
    print('\n')

    ## Extracts information from metatags and incoporates to the plain text
    metatext.append(textweb)
    metatext = [x.replace('\n', ' ') for x in metatext]
    text = ' '.join(map(str, metatext))
    print('Final Text: ')
    print(text)
    print('\n')

    '''        PreProcessing     '''
    ## Tokenization
    tokens = nltk.word_tokenize(text)
    print('TOKENS: ')
    print(tokens)

    ## Removing tokens that has the following symbols
    wordsFiltered = []
    symbols = ['dr','Dr','@', '/', '©']
    for i in tokens:
        if not i in symbols:
            wordsFiltered.append(i)

    ## Lemmanizing
    wnl = nltk.WordNetLemmatizer()
    lem = [wnl.lemmatize(t) for t in wordsFiltered]
    print('LEMANIZING: ')
    print(lem)
    print('\n')

    '''       POS - Tagging  & Entity Recognition (NER)      '''
    ## Adding pos-tags to tokens
    tags = nltk.pos_tag(lem)
    print('Name Entity Recognition: ')
    print(tags)
    print('\n')

    ## Adding NER to tokens
    ententy = nltk.ne_chunk(tags)
    print('POS-TAG: ')
    print(ententy)
    print('\n')

    ## Grammar Rules applied
    grammar = """  Name: { < NNP > < NNP >}
                         { < NNP > < IN > < NNP >}
                         { < NN.*|JJ>*<NN.*>}  """

    chunker = nltk.RegexpParser(grammar)
    parse = chunker.parse(tags)
    print('Grammar Results: ')
    print(parse)
    print(type(parse))
    print('\n')

    '''--------- Post-Processing or Normalization ---------'''
    ## Sentence tagging and list creation
    sen = nltk.chunk.tree2conlltags(parse)
    print('Sentence Tagging: ')
    print(sen)
    print('\n')

    # Stemming and lowering the case of words
    sno = nltk.SnowballStemmer("english")
    list = []
    a = []
    for i in range(0, len(sen)):
        if sen[i][2] == 'B-Name':
            if len(a) == 0:
                a.append(sno.stem(sen[i][0].lower()))
            else:
                list.append(' '.join(a))
                a.clear()
                a.append(sno.stem(sen[i][0].lower()))
        if sen[i][2] == 'I-Name':
            a.append(sno.stem(sen[i][0].lower()))
        if sen[i][2] == 'O':
            list.append(' '.join(a))
            list.append(sno.stem(sen[i][0].lower()))
            a.clear()

    listfiltered = []

    ## Removing funy characters, stopwords, punctuation and digists
    wordsFiltered = []
    symbols = ['``', '...', "''", '/', '!', "'", ':', '+', '=']
    stopWords = set(stopwords.words('english'))

    for i in list:
        if not (i in symbols or i in stopWords or i in string.punctuation or i.isdigit()):
            listfiltered.append(i)

    ListWords.append(listfiltered)
    print('Final List of words for url "i"')
    print(sen)
    print('\n')

print('Total list of words before indexation: ')
for i in ListWords:
    print(i)

'''--------- Ranking ---------'''

## convert list of list into a flat list
flat_list = [word for i in ListWords for word in i]

## Creates a list of non duplicated words
vocabulary = set(flat_list)
print('Dictionary: ')
print(vocabulary)
print('\n')

## Defines N and vocabulary size
vocabulary_size = len(vocabulary)
documents = len(url)
print('vocabulary_Size: ',vocabulary_size)
print('Document Size: ',documents)
print('\n')

## Generates idf with  the formula log(N/idf)
idf = collections.defaultdict(int)
docs = collections.defaultdict(str)
# print(ListWords[0][:]) # word -> udo kruschwitz  (first document)
# print(ListWords[1][:]) # word - > irsg             (second document)

## creates idf iterator
for doc in range(0, len(url)):
    words = set(ListWords[doc][:])
    for word in words:
        idf[word] += 1
        docs[word] += str(doc + 1) + ', '
    words.clear()

## Calculating idf
for word in vocabulary:
    idf[word] = math.log(documents / float(idf[word]))

## Creates tf iterator per document
for i in range(0,len(url)):
    tf = collections.defaultdict(int)
    words = ListWords[doc][:]
    for word in words:
        tf[word] += 1

    ## Calculating tf.idf for URL "i"
    tf_idf = collections.defaultdict(int)
    for word in vocabulary:
        tf_idf[word] = tf[word] * idf[word]

    ## Printing tf.idf for URL "i"
    TF_IDF = collections.OrderedDict(sorted(tf_idf.items(), key=itemgetter(1), reverse=True))
    print('TF.IDF TABLE FOR URL:   ', url[i])
    df = pd.DataFrame([TF_IDF]).T
    df.columns = ['TF.IDF']
    df.index.name = 'DICTIONARY OF WORDS'
    print(df)


'''         End of the code                      '''