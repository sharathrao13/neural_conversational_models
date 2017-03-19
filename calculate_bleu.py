import nltk
import numpy as np

def evaluate_bleu_2():

    chat_responses = []
    references = []

    file_chat = open("responses.txt")
    file_refer = open("actual_responses.txt")

    maxlen = 8000
    for i in range(maxlen):
        chat = file_chat.readline()
        refer = file_refer.readline()

        if chat.lower() == "I don't know".lower():
            continue

        c = chat.split(" ")
        r = refer.split(" ")
        if len(c) < 2 or len(r) <2:
            continue
        chat_responses.append(c)
        references.append(r)
    file_chat.close()
    file_refer.close()

    plot_list = []
    bleu_scores = []
    for i in range(len(chat_responses)):
        score = nltk.translate.bleu_score.sentence_bleu([references[i]], chat_responses[i],weights=(0.5,0.5))
        bleu_scores.append(score)
        plot_list.append([len(" ".join(chat_responses[i])),score])

    print plot_list[:10]
    sorted_list = sorted(plot_list,key=lambda x:x[0])
    plot_file = open("plot_data_bleu_2","wb")
    for data in sorted_list:
        val = str(data[0])+" "+"%0.2f"%data[1] + "\n"
        plot_file.write(val)

    plot_file.close()

    print "The average bleu score is %s"%(np.mean(bleu_scores))
    print "The maximum bleu score is %s"%(max(bleu_scores))

def evaluate_bleu_4():

    chat_responses = []
    references = []

    file_chat = open("responses.txt")
    file_refer = open("actual_responses.txt")

    maxlen = 8000
    for i in range(maxlen):
        chat = file_chat.readline()
        refer = file_refer.readline()

        if chat.lower() == "I don't know".lower():
            continue

        c = chat.split(" ")
        r = refer.split(" ")
        if len(c) < 4 or len(r) <4:
            continue
        chat_responses.append(c)
        references.append(r)
    file_chat.close()
    file_refer.close()

    bleu_scores = []
    plot_list = []
    for i in range(len(chat_responses)):
        score = nltk.translate.bleu_score.sentence_bleu([references[i]], chat_responses[i],weights=(0.25,0.25,0.25,0.25))
        bleu_scores.append(score)
        plot_list.append([len(" ".join(chat_responses[i])),score])

    print plot_list[:10]

    sorted_list = sorted(plot_list,key=lambda x:x[0])

    plot_file = open("plot_data_bleu_4","wb")
    for data in sorted_list:
        val = str(data[0])+" "+"%0.2f"%data[1] + "\n"
        plot_file.write(val)

    plot_file.close()

    print "The average bleu score is %s"%(np.mean(bleu_scores))
    print "The maximum bleu score is %s"%(max(bleu_scores))


evaluate_bleu_2()
print "-------------------------------"
evaluate_bleu_4()