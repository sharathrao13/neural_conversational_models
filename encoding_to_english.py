import nltk
import numpy as np

def create_val_engish():
    vocab_path = './vocabulary'
    file = open("y_dev.txt")
    sentence = file.readline()
    actual_responses = []
    while sentence:
        s = sentence.split(" ")
        r =[int(x) for x in s if x]
        res = " ".join([str(rev_vocab[word]) for word in r])
        actual_responses.append(res)
        sentence = file.readline()
    file.close()

    file_r = open("actual_responses.txt","wb")
    for i in range(len(actual_responses)):
        file_r.write(actual_responses[i])
        file_r.write("\n")
    file_r.close()