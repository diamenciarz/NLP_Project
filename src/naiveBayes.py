from cmath import log
import re
import string

import csv

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

import numpy as np

import json


def process_abs(abs):
    '''
    Input:
        abs: a string the abstract
    Output:
        abs_clean: a list of words containing the processed abstract

    '''
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    # remove punctuation
    abstract = re.sub(r"[^a-zA-Z0-9]", " ", abs)
    # remove numbers
    abstract = re.sub(r"\d+", " ", abs)
   
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    abs_tokens = tokenizer.tokenize(abstract)

    abs_clean = []
    for word in abs_tokens:
        if (word not in stopwords_english and  # remove stopwords
            word not in string.punctuation):  # remove punctuation
            # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            abs_clean.append(stem_word)

    return abs_clean


def count_abs(result, abs, labels):
    '''
    Input:
        result: a dictionary that will be used to map each pair to its frequency
        abs: a list of abstract
        labels:  A list corresponding to the scientific branch label of each abstract 
    Output:
        result: a dictionary mapping each pair to its frequency
    '''

    for label, abs in zip(labels, abs):
        for word in process_abs(abs):
            # define the key, which is the word and label tuple
            pair = (word,label)

            # if the key exists in the dictionary, increment the count
            if pair in result:
                result[pair] += 1

            # else, if the key is new, add it to the dictionary and set the count to 1
            else:
                result[pair] = 1

    return result

import json

def label_data(json_data):
    # Opening JSON file
    #f = open('C:\\Users\\ruben\\Downloads\\joder\\arxiv-metadata-oai-snapshot.json')
    
    # returns JSON object as 
    # a dictionary
    data = json_data
    
    # Iterating through the json
    # list
    #for i in data['categories']:
   # print(data['categories'])
    str = data['categories']
    #print(catList)

    

    
    # Closing file
   # f.close()

def check_category(category_string):
    category_list = category_string.split()
    catList = []
    for string in category_list:
        
        if cs(string): 
            catList.append(0)
        if electSyst(string):
            catList.append(1)
        if math(string):
            catList.append(2)
        if phy(string):
            catList.append(3)
        if bio(string):
            catList.append(4)
        if fin(string):
            catList.append(5)
        if stat(string):
            catList.append(6)
        if econm(string):
            catList.append(7)
    
    return catList
    

def cs(str):
#computer science
    return ('cs.' in str and str.find('cs.') == 0)

def econm(str):
# economics
    return ('econ.EM' in str or 'econ.GN' in str or 'econ.TH' in str)
def electSyst(str):
# Electrical Engineering and Sustems Science
    return ('eess.' in str and str.find('eess.') == 0)
def math(str):
# Mathematics
    return ('math.' in str and str.find('math.') == 0)
def phy(str):
    return ('-ph.' in str or 'cond-mat' in str or 'gr-qc' in str or 'hep' in str or 'nlin' in str or 'nucl' in str or 'physics' in str )
def bio(str):
    return('q-bio' in str)
def fin(str):
    return 'q-fin' in str
def stat(str):
    return 'stat.' in str

token_map = {}

def labelData(token_list, category):
    for c in category:
        if c in countInstancesperClass:
            countInstancesperClass[c] += 1
        else:
            countInstancesperClass[c] = 1
    # label the tokens  from the abs with their category
    weight = 1
    for token in token_list:
        
        # check if token is in the hashmap
        
        if token not in token_map:
            token_map[token] = {}  # Create an empty inner dictionary for the token key
            inner_dict = token_map[token]
        # You can add/update the weight for a specific category
            
              # Replace with the actual weight value
            for c in category:
                inner_dict[c] = weight
                print(token + ' '+ str(c) + ' '+ str(weight) )
    # check if the category is already in the token hashmap
        else:
            inner_dict = token_map[token]
            for c in category:
                
                if  c in inner_dict:
                        inner_dict[c] += 1
                        
                
                else:
                    inner_dict[c] = weight
                print(token + ' '+ str(c)+ ' '+ str(inner_dict[c]))



def startLabel():
    
            data = getData_homogeneuos()
            for pair in data:  
                # Process the JSON object
                print(pair[0])
                labelData(process_abs(pair[1]), pair[0])
                write_dict_to_csv(token_map,"C:\\Users\\ruben\\Downloads\\NLP_Project-main\\words-label.csv")

                
def write_dict_to_csv(dictionary, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Word', 'Label', 'Weight'])
        for word, inner_dict in dictionary.items():
            if inner_dict:
                max_value = max(inner_dict.values())
                max_key = max(inner_dict, key=lambda k: inner_dict[k])
                print(word, max_key, max_value)
                writer.writerow([word, inner_dict.keys(), inner_dict.values()])

def getData_homogeneuos():
    abs_list = []
    category_tracker = {}
    category_tracker[0] = 0
    category_tracker[1] = 0
    category_tracker[2] = 0
    category_tracker[3] = 0
    category_tracker[4] = 0
    category_tracker[5] = 0
    category_tracker[6] = 0
    category_tracker[7] = 0


    with open('C:\\Users\\ruben\\Downloads\\NLP_Project-main\\arxiv-metadata-oai-snapshot.json', 'r') as file:
    

        # Iterate over each line in the file
        counter = 0
        for line in file:
            counter += 1
            print(counter)
            
            json_obj = json.loads(line)
            category = check_category(json_obj['categories'])
            abstract = json_obj['abstract']
            for i in category:
                if category_tracker[i] <= 200:
                    abs_list.append([category, abstract])
                    for c in category:
                        category_tracker[c] += 1
                    break  
                    
    with open('C:\\Users\\ruben\\Downloads\\NLP_Project-main\\data.json', 'w') as json_file:
        for pair in abs_list:
            json_obj = {
            'abstract': pair[1],
            'category': pair[0]
        }
            json.dump(json_obj, json_file)
            json_file.write('\n')
        json_file.close()  # Close the file


        return abs_list    

def train_naive_bayes():
    """
    Train a Naive Bayes classifier for multiclass classification.

    Args:
        label_data: List of training data examples.

    Returns:
        logprior: Dictionary of log priors for each category.
        loglikelihood: Dictionary of log likelihoods for each token and category.
    """
    logprior = {}
    loglikelihood = {}
    for i in range(8):

        if getCountClass()[i]:
            logprior[i] = log(getCountClass()[i]  / 200 )
        else:
            logprior[i] = 0
    for i in token_map:
        with open('C:\\Users\\ruben\\Downloads\\NLP_Project-main\\words.json', 'r') as file:
            for line in file:
                for line in file:
                    json_obj = json.loads(line)
                    for j in len(json_obj[i]['category']):
                        loglikelihood[i, json_obj[i]['category'][j]] = log( json[i]['appearances']+1/ countInstancesperClass[json_obj[i]['category'][j]]+  len(token_map))


    return logprior,loglikelihood

countInstancesperClass = {}


def getSmallestClass(clountclass, classes):
    classesValue = {}
    for i in classes:
        classesValue[i] = clountclass[i]

    return min(classesValue, key=lambda k: classesValue[k])


def predict_naive_bayes(abstract_tokens, logprior, loglikelihood):
    class_scores = {}
    with open('C:\\Users\\ruben\\Downloads\\NLP_Project-main\\words.json', 'r') as file:
        for line in file:
            json_obj = json.loads(line)
            for i in range(8):
            
                    class_scores[i] = logprior[i]

            for i in range(8):
                for j in abstract_tokens:
                    if loglikelihood.get((j, i)):
                        class_scores[i] += loglikelihood[(j, i)]
                        complex_num = class_scores[i]
                        real_part = complex_num.real 
                        class_scores[i] = real_part

    print(class_scores)
    predicted_class = max(class_scores, key=lambda k: class_scores[k].real)
    return predicted_class



    
def getCountClass():
    categCOunt = [0,0,0,0,0,0,0,0]
    with open('C:\\Users\\ruben\\Downloads\\NLP_Project-main\\src\\naiveBayes\\data.json', 'r') as file:
    

        # Iterate over each line in the file
        
        for line in file:
            json_obj = json.loads(line)
            if(0 in json_obj['category']):
                categCOunt[0] += 1

            if(1 in json_obj['category']):
                categCOunt[1] += 1
            
            if(2 in json_obj['category']):
                categCOunt[2] += 1
            
            if(3 in json_obj['category']):
                categCOunt[3] += 1
            
            if(4 in json_obj['category']):
                categCOunt[4] += 1
            
            if(5 in json_obj['category']):
                categCOunt[5] += 1
        
            if(6 in json_obj['category']):
                categCOunt[6] += 1
            
            if(7 in json_obj['category']):
                categCOunt[7] += 1
    return categCOunt
def transform_csv_to_json(csv_file, json_file):
    result_dict = {}

    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)

        # Skip header row if present
        next(csv_reader)

        for row in csv_reader:
            token = row[0]
            category = row[1]
            appearances = int(row[2])

            if token not in result_dict:
                result_dict[token] = {"category": category, "appearances": appearances}

    with open(json_file, 'w') as file:
        json.dump(result_dict, file)
    
#startLabel()
print(getCountClass)
abstract = 'Computer Science is the study of computers and computational systems. Unlike electrical and computer engineers, computer scientists deal mostly with software and software systems; this includes their theory, design, development, and application.Principal areas of study within Computer Science include artificial intelligence, computer systems and networks, security, database systems, human computer interaction, vision and graphics, numerical analysis, programming languages, software engineering, bioinformatics and theory of computing.Although knowing how to program is essential to the study of computer science, it is only one element of the field. Computer scientists design and analyze algorithms to solve programs and study the performance of computer hardware and software. The problems that computer scientists encounter range from the abstract-- determining what problems can be solved with computers and the complexity of the algorithms that solve them – to the tangible – designing applications that perform well on handheld devices, that are easy to use, and that uphold security measures.Graduates of University of Maryland’s Computer Science Department are lifetime learners; they are able to adapt quickly with this challenging field.'
logprior, loglikelihood = train_naive_bayes()
print(predict_naive_bayes(process_abs(abstract), logprior, loglikelihood))

print(getCountClass())
# import json

# label_data = []

# label_data = []
# with open('C:\\Users\\ruben\\Downloads\\NLP_Project-main\\arxiv-metadata-oai-snapshot.json', 'r') as f:
#     for line in f:
#         data = json.loads(line)
#         label_data.append(data)


# logprior, loglikelihood = train_naive_bayes(label_data)
# print(logprior)
