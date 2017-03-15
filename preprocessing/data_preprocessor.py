import re
import collections

vocabulary_size = 10000
vocabulary_path = "../cache/vocabulary"

X_train_path = 'cache/X_train'
y_train_path = 'cache/y_train'
X_val_path = 'cache/X_val'
y_val_path = 'cache/y_val'

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

_WORD_SPLIT = re.compile(b"([.,!?\":;)(])")

def create_save_dictionary(words, vocabulary_path, vocabulary_size):

    count = [['_UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))

    dictionary = dict()

    for word, _ in count:
        dictionary[word] = len(dictionary)

    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    f = open(vocabulary_path, 'w')
    for key in dictionary:
        f.write(key + '\n')
    f.close()

    return dictionary, reverse_dictionary

def generate_encoded_files(tokenized_sentences, dictionary):

    encoded_holder = []
    f1 = open(X_train_path, 'w')

    last_line = tokenized_sentences.pop()
    first_line = tokenized_sentences.pop(0)
    dev_counter = int(len(tokenized_sentences) - len(tokenized_sentences) / fraction_dev)

    unk_id = dictionary['_UNK']
    first_line_encoded = encode_sentence(first_line, dictionary, unk_id)
    f1.write(first_line_encoded + '\n')

    for x in xrange(dev_counter):
        encoded_sentence = encode_sentence(tokenized_sentences[x], dictionary, unk_id)
        encoded_holder.append(encoded_sentence)
        f1.write(encoded_sentence + '\n')  # Write sentence to file
    f1.close()

    d1 = open(X_val_path, 'w')
    for x in xrange(dev_counter, len(tokenized_sentences)):
        encoded_sentence = encode_sentence(tokenized_sentences[x], dictionary, unk_id)
        encoded_holder.append(encoded_sentence)
        d1.write(encoded_sentence + '\n')  # Write sentence to file

    d1.close()

    f2 = open(y_train_path, 'w')

    for x in xrange(dev_counter + 1):
        f2.write(encoded_holder[x] + '\n')  # Write sentence to file

    f2.close()

    d2 = open(y_val_path, 'w')
    for x in xrange(dev_counter + 1, len(tokenized_sentences)):
        d2.write(encoded_holder[x] + '\n')  # Write sentence to file

    last_line_encoded = encode_sentence(last_line, dictionary, unk_id)
    d2.write(last_line_encoded + '\n')
    d2.close()

def encode_sentence(sentence, dictionary, unk_id):
    if not sentence:
        return ""
    first_word = sentence.pop(0)
    if first_word in dictionary:
        encoded_sentence = str(dictionary[first_word])
    else:
        encoded_sentence = str(unk_id)

    for word in sentence:
        if word in dictionary:
            encoded_word = dictionary[word]
        else:
            encoded_word = unk_id
        encoded_sentence += " " + str(encoded_word)
    return encoded_sentence

def read_data(path, read_as_sentence = False):
    data = []
    lines = [line.rstrip('\n') for line in open(path)]
    local_data = []

    if read_as_sentence:
        for line in lines:
            local_data.append(re.findall(r'\S+', line))
    else:
        for line in lines:
            local_data.extend(re.findall(r'\S+', line))

    data.extend(local_data)
    return data

def encode_test_sentence(sentence, vocabulary):
    words = []

    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(_WORD_SPLIT, space_separated_fragment))

    words_clean = [w for w in words if w]

    return [vocabulary.get(w, UNK_ID) for w in words_clean]

def load_vocabulary(vocabulary_path):
    rev_vocab = []
    with open(vocabulary_path, mode="rb") as f:
        rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab

def prepare_dataset_encoded(dataset_path, vocabulary_size):

    data = read_data(dataset_path)
    dictionary, reverse_dictionary = create_save_dictionary(data, vocabulary_path, vocabulary_size)
    sentences = read_data(dataset_path, read_as_sentence=True)
    generate_encoded_files(sentences,dictionary)
