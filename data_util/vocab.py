import hashlib

import array
from collections import Counter

import logging
import torch

logger = logging.getLogger(__name__)


"""
This is a util module for:
    1. Loading word embeddings.
    2. Building vocabulary for your corpus.
"""



class STOI(dict):
    def __init__(self, unk_num=1, **kwargs):
        super().__init__(**kwargs)
        self.unk_num = unk_num

    def __getitem__(self, k):
        if self.unk_num == 0 or k in super().keys():
            return super().__getitem__(k)
        else:
            k = STOI.hash_string(k, self.unk_num)
            return super().__getitem__(k)

    @staticmethod
    def hash_string(input_str, unk_num):
        hcode = int(hashlib.sha1(input_str.encode('utf-8')).hexdigest(), 16)
        hcode %= unk_num
        return "<unk-" + str(hcode) + ">"


class Embedding(object):
    def __init__(self, file_name, cache_file_name=None,
                 encoding='utf-8',
                 unk_init=torch.Tensor.zero_):
        """Arguments:
               file_name: name of the file that contains the embeddings
               cache_file_name: directory for cached embeddings
               unk_init (callback): by default, initalize out-of-vocabulary word vectors
                   to zero vectors; can be any function that takes in a Tensor and
                   returns a Tensor of the same size
         """
        self.unk_init = unk_init
        self.cache(file_name, cache_file_name, encoding)

    def __getitem__(self, token):
        if token in self.stoi:
            return self.vectors[self.stoi[token]]
        else:
            return self.unk_init(torch.Tensor(1, self.dim))

    def cache(self, file_name, cache_file_name, encoding):
        """
        :param file_name: the original file that contains the WordEmbedding.
        :param cache_file_name: the name of the cache_file. (usually with .pt)
        :param encoding: encoding of the embedding file.
        :return: 
        """

        if not cache_file_name:
            cache_file_name = os.path.splitext(file_name)[0] + ".pt"

        if not os.path.isfile(cache_file_name):
            itos, vectors, dim = [], array.array(str('d')), None
            # Try to read the whole file with utf-8 encoding.
            binary_lines = False
            try:
                with open(file_name, encoding=encoding) as f:
                    lines = [line for line in f]
            # If there are malformed lines, read in binary mode
            # and manually decode each word from utf-8
            except:
                logger.warning("Could not read {} as UTF8 file, "
                               "reading file as bytes and skipping "
                               "words with malformed UTF8.".format(file_name))
                with open(file_name, 'rb') as f:
                    lines = [line for line in f]
                binary_lines = True

            logger.info("Loading vectors from {}".format(file_name))
            for line in tqdm(lines, total=len(lines)):
                # Explicitly splitting on " " is important, so we don't
                # get rid of Unicode non-breaking spaces in the vectors.
                entries = line.rstrip().split(" ")
                word, entries = entries[0], entries[1:]
                if dim is None and len(entries) > 1:
                    dim = len(entries)
                elif len(entries) == 1:
                    logger.warning("Skipping token {} with 1-dimensional "
                                   "vector {}; likely a header".format(word, entries))
                    continue
                elif dim != len(entries):
                    raise RuntimeError(
                        "Vector for token {} has {} dimensions, but previously "
                        "read vectors have {} dimensions. All vectors must have "
                        "the same number of dimensions.".format(word, len(entries), dim))

                if binary_lines:
                    try:
                        if isinstance(word, six.binary_type):
                            word = word.decode('utf-8')
                    except:
                        logger.info("Skipping non-UTF8 token {}".format(repr(word)))
                        continue
                vectors.extend(float(x) for x in entries)
                itos.append(word)

            self.itos = itos
            self.stoi = {word: i for i, word in enumerate(itos)}
            self.vectors = torch.Tensor(vectors).view(-1, dim)
            self.dim = dim
            logger.info('Saving vectors to {}'.format(cache_file_name))
            torch.save((self.itos, self.stoi, self.vectors, self.dim), cache_file_name)
        else:
            logger.info('Loading vectors from {}'.format(cache_file_name))
            self.itos, self.stoi, self.vectors, self.dim = torch.load(cache_file_name)


class ExVocab(object):
    """Defines a vocabulary object that will be used to numericalize a field.
    Attributes:
        freqs: A collections.Counter object holding the frequencies of tokens
            in the data used to build the Vocab.
        stoi: A collections.defaultdict instance mapping token strings to
            numerical identifiers.
        itos: A list of token strings indexed by their numerical identifiers.
    """
    def __init__(self, max_size=None, min_freq=1,
                 init_elements_list=['<pad>', '<sos>', '<eos>'],
                 unk_num=1):
        """Create a Vocab object from a collections.Counter.
        Arguments:
            counter: collections.Counter object holding the frequencies of
                each value found in the data.
            max_size: The maximum size of the vocabulary, or None for no
                maximum. Default: None.
            min_freq: The minimum frequency needed to include a token in the
                vocabulary. Values less than 1 will be set to 1. Default: 1.
            init_elements_list: The list of initial element list. Default: ['<pad>']
            unk_num: Number of unknown token to be hashed.
        """
        self.freqs = Counter()
        self.vectors = None
        self.init_list = init_elements_list
        self.freqs.update(self.init_list)
        self.unk_num = unk_num

        min_freq = max(min_freq, 1)

        self.itos = self.init_list

        self.stoi = STOI(unk_num=unk_num)
        self.stoi.update({tok: i for i, tok in enumerate(self.init_list)})

        self.max_size = None if max_size is None else max_size + len(self.itos)
        self.min_freq = min_freq

    def popularize_corpus(self, counters, addition_list=[], delete_list=[]):
        global_counter = Counter()

        if not isinstance(counters, list):
            counters = [counters]
        for counter in counters:
            global_counter.update(counter)

        global_counter.subtract({tok: self.freqs[tok] for tok in self.init_list})
        global_counter.subtract({tok: self.freqs[tok] for tok in delete_list})

        global_counter.update(addition_list)

        for i in range(self.unk_num):
            unk_name = "<unk-{0}>".format(i)
            if unk_name in global_counter:
                del global_counter[unk_name]

        # Sort by frequency, then alphabetically
        words_and_frequencies = sorted(global_counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for word, freq in words_and_frequencies:
            if freq < self.min_freq or len(self.itos) == self.max_size:
                break
            self.itos.append(word)
            self.stoi[word] = len(self.itos) - 1

        self.freqs.update(global_counter)

        # Adding <unk-[#s]> to the vocabulary.
        for i in range(self.unk_num):
            new_unk_entry = '<unk-{}>'.format(i)
            self.itos.append(new_unk_entry)
            self.stoi[new_unk_entry] = len(self.itos) - 1

    def popularize_embedding(self, embeddings: Embedding, unk_init=None):
        dim = embeddings.dim
        self.vectors = torch.Tensor(len(self), dim)

        for i, token in enumerate(self.itos):
            if token in embeddings.stoi:
                self.vectors[i] = embeddings[token]
            else:
                if unk_init is None:
                    random_init = torch.Tensor(dim).uniform_(-0.01, 0.01)
                    self.vectors[i] = random_init
                else:
                    self.vectors[i] = unk_init(self.vectors[i])

        # Set "<pad>" to all zeros
        self.vectors[self.stoi['<pad>']] = 0

    def cache(self, cache_file_name):
        logger.info('Saving vectors to {}'.format(cache_file_name))
        torch.save((self.itos, self.stoi, self.freqs, self.vectors), cache_file_name)

    def load(self, cached_file_name):
        logger.info('Loading vectors from {}'.format(cached_file_name))
        self.itos, self.stoi, self.freqs, self.vectors = torch.load(cached_file_name)

    def __eq__(self, other):
        if self.freqs != other.freqs:
            return False
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        if self.vectors != other.vectors:
            return False
        return True

    def __len__(self):
        return len(self.itos)