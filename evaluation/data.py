import re
import nltk
import string
import pandas as pd
import random
from tqdm import tqdm

from typing import List, Tuple, Union
from octis.dataset.dataset import Dataset
from octis.preprocessing.preprocessing import Preprocessing

#load library
import pandas as pd
import numpy as np
import gensim
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem.porter import *
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
import re
import nltk
from nltk.tokenize import word_tokenize
from language_detector import detect_language
import pkg_resources
from symspellpy import SymSpell, Verbosity
import en_core_web_sm
from sklearn.feature_extraction.text import TfidfVectorizer
from octis.dataset.dataset import Dataset

nltk.download("punkt", download_dir='/mnt/raid1/gzhou/anaconda3/envs/evanew/nltk_data')
# nltk.download(download_dir="/mnt/raid1/gzhou", info_or_id="punkt")
# nltk.download("punkt")


class DataLoader:
    """Prepare and load custom data using OCTIS

    Arguments:
        dataset: The name of the dataset, default options:
                    * trump
                    * 20news

    Usage:

    **Trump** - Unprocessed

    ```python
    from evaluation import DataLoader
    dataloader = DataLoader(dataset="trump").prepare_docs(save="trump.txt").preprocess_octis(output_folder="trump")
    ```

    **20 Newsgroups** - Unprocessed

    ```python
    from evaluation import DataLoader
    dataloader = DataLoader(dataset="20news").prepare_docs(save="20news.txt").preprocess_octis(output_folder="20news")
    ```

    **Custom Data**

    Whenever you want to use a custom dataset (list of strings), make sure to use the loader like this:

    ```python
    from evaluation import DataLoader
    dataloader = DataLoader(dataset="my_docs").prepare_docs(save="my_docs.txt", docs=my_docs).preprocess_octis(output_folder="my_docs")
    ```
    """

    def __init__(self, dataset: str):
        self.dataset = dataset
        self.docs = None
        self.timestamps = None
        self.octis_docs = None
        self.doc_path = None
        
        self.sym_spell = SymSpell(max_dictionary_edit_distance=3, prefix_length=7)
        dictionary_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_dictionary_en_82_765.txt")
        if self.sym_spell.word_count:
            pass
        else:
            self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
            
        self.nlp = en_core_web_sm.load(disable=['parser', 'ner'])

    def load_docs(
        self, save: bool = False, docs: List[str] = None
    ) -> Tuple[List[str], Union[List[str], None]]:
        """Load in the documents

        ```python
        dataloader = DataLoader(dataset="trump")
        docs, timestamps = dataloader.load_docs()
        ```
        """
        if docs is not None:
            return self.docs, None

        if self.dataset == "20news":
            self.docs, self.timestamps = self._20news()
        elif self.dataset == "agnews":
            self.docs, self.timestamps = self._agnews()
        elif self.dataset == "yelp":
            self.docs, self.timestamps = self._yelp()
        elif self.dataset == "yelp_full":
            self.docs, self.timestamps = self._yelp_full()
        elif self.dataset == "dbpedia":
            self.docs, self.timestamps = self._dbpedia()
        elif self.dataset == "dbpedia_full":
            self.docs, self.timestamps = self._dbpedia_full()
        elif self.dataset == "StackOverflow":
            self.docs, self.timestamps = self._StackOverflow()
        elif self.dataset == "yahoo_answer":
            self.docs, self.timestamps = self._yahoo_answer()

        if save:
            self._save(self.docs, save)
        
        return self.docs, self.timestamps

    def load_octis(self, custom: bool = False) -> Dataset:
        """Get dataset from OCTIS

        Arguments:
            custom: Whether a custom dataset is used or one retrieved from
                    https://github.com/MIND-Lab/OCTIS#available-datasets

        Usage:

        ```python
        from evaluation import DataLoader
        dataloader = DataLoader(dataset="20news")
        data = dataloader.load_octis(custom=True)
        ```
        """
        data = Dataset()

        if custom:
            data.load_custom_dataset_from_folder(self.dataset)
        else:
            data.fetch_dataset(self.dataset)

        self.octis_docs = data
        return self.octis_docs

    def prepare_docs(self, save: bool = False, docs: List[str] = None):
        """Prepare documents

        Arguments:
            save: The path to save the model to, make sure it ends in .json
            docs: The documents you want to preprocess in OCTIS

        Usage:

        ```python
        from evaluation import DataLoader
        dataloader = DataLoader(dataset="my_docs").prepare_docs(save="my_docs.txt", docs=my_docs)
        ```
        """
        self.load_docs(save, docs)
        return self

    def preprocess_octis(
        self,
        preprocessor: Preprocessing = None,
        documents_path: str = None,
        output_folder: str = "docs",
        sentence_need=False,
    ):
        """Preprocess the data using OCTIS

        Arguments:
            preprocessor: Custom OCTIS preprocessor
            documents_path: Path to the .txt file
            output_folder: Path to where you want to save the preprocessed data

        Usage:

        ```python
        from evaluation import DataLoader
        dataloader = DataLoader(dataset="my_docs").prepare_docs(save="my_docs.txt", docs=my_docs)
        dataloader.preprocess_octis(output_folder="my_docs")
        ```

        If you want to use your custom preprocessor:

        ```python
        from evaluation import DataLoader
        from octis.preprocessing.preprocessing import Preprocessing

        preprocessor = Preprocessing(lowercase=False,
                                remove_punctuation=False,
                                punctuation=string.punctuation,
                                remove_numbers=False,
                                lemmatize=False,
                                language='english',
                                split=False,
                                verbose=True,
                                save_original_indexes=True,
                                remove_stopwords_spacy=False)

        dataloader = DataLoader(dataset="my_docs").prepare_docs(save="my_docs.txt", docs=my_docs)
        dataloader.preprocess_octis(preprocessor=preprocessor, output_folder="my_docs")
        ```
        """
        # if preprocessor is None:
        #     preprocessor = Preprocessing(
        #         lowercase=True,
        #         remove_punctuation=True,
        #         punctuation=string.punctuation,
        #         remove_numbers=True,
        #         lemmatize=True,
        #         language="english",
        #         split=False,
        #         verbose=True,
        #         save_original_indexes=True,
        #         remove_stopwords_spacy=True,
        #         min_chars = 3,
        #         min_df = 0.0075,
        #         max_df= 0.95,
        #     )
        if not documents_path:
            documents_path = self.doc_path
        # dataset = preprocessor.preprocess_dataset(documents_path=documents_path)
        if not sentence_need:
            dataset = self._preprocess_dataset_custom(documents_path=documents_path)
        else:
            dataset, sents = self._preprocess_dataset_custom(documents_path=documents_path, sentence_on=sentence_need)
            self._save(sents, f'sentences_{self.dataset}')
        
        dataset.save(output_folder)

    def _20news(self) -> Tuple[List[str], List[str]]:
        """Prepare the trump dataset"""
        from sklearn.datasets import fetch_20newsgroups
        newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
        docs = pd.Series(newsgroups_train.data).to_list()
        docs = [doc.lower().replace("\n", " ") for doc in docs if len(doc) > 2]
        return docs, None

    def _agnews(self) -> Tuple[List[str], List[str]]:
        """Prepare the trump dataset"""
        agnews = pd.read_csv('/mnt/raid1/gzhou/datasets/agnews/ag_news_csv/train.csv', header=None)
        docs = agnews[2].to_list()
        docs = [doc.lower().replace("\n", " ") for doc in docs if len(doc) > 2]
        return docs, None
    
    def _yelp(self) -> Tuple[List[str], List[str]]:
        """Prepare the trump dataset"""
        yelp = pd.read_csv('/mnt/raid1/gzhou/datasets/yelp/yelp_review_full_csv/train.csv', header=None)
        df = pd.DataFrame()
        y = list(range(0, 650000))
        slice_dbpedia = random.sample(y, 120000)
        slice_dbpedia = sorted(slice_dbpedia)
        for num in tqdm(range(120000)):
            df = df.append(yelp.loc[slice_dbpedia[num]])
        df.reset_index(drop=True, inplace=True)
        docs = df[1].to_list()
        # docs = yelp[1].to_list()
        docs = [doc.lower().replace("\n", " ") for doc in docs if len(doc) > 2]
        return docs, None

    def _dbpedia(self) -> Tuple[List[str], List[str]]:
        dbpedia = pd.read_csv('/mnt/raid1/gzhou/datasets/dbpedia/dbpedia_csv/train.csv', header=None)
        df = pd.DataFrame()
        y = list(range(0, 560000))
        slice_dbpedia = random.sample(y, 120000)
        slice_dbpedia = sorted(slice_dbpedia)
        for num in tqdm(range(120000)):
            df = df.append(dbpedia.loc[slice_dbpedia[num]])
        df.reset_index(drop=True, inplace=True)
        docs = df[2].to_list()
        # docs = dbpedia[2].to_list()
        docs = [doc.lower().replace("\n", " ") for doc in docs if len(doc) > 2]
        return docs, None
    
    def _yelp_full(self) -> Tuple[List[str], List[str]]:
        """Prepare the trump dataset"""
        yelp = pd.read_csv('/mnt/raid1/gzhou/datasets/yelp/yelp_review_full_csv/train.csv', header=None)
        docs = yelp[1].to_list()
        docs = [doc.lower().replace("\n", " ") for doc in docs if len(doc) > 2]
        return docs, None

    def _dbpedia_full(self) -> Tuple[List[str], List[str]]:
        dbpedia = pd.read_csv('/mnt/raid1/gzhou/datasets/dbpedia/dbpedia_csv/train.csv', header=None)
        docs = dbpedia[2].to_list()
        docs = [doc.lower().replace("\n", " ") for doc in docs if len(doc) > 2]
        return docs, None
    
    def _reuters_21578(self) -> Tuple[List[str], List[str]]:
        dbpedia = pd.read_csv('/mnt/raid1/gzhou/datasets/reuters21578/reuters-21578-json/reuters_21578.csv', header=None)
        docs = dbpedia[0].to_list()
        docs = [doc.lower().replace("\n", " ") for doc in docs if len(doc) > 2]
        return docs, None
    
    def _StackOverflow(self) -> Tuple[List[str], List[str]]:
        dbpedia = pd.read_csv('/mnt/raid1/gzhou/datasets/StackOverflow/train.csv', header=None)
        docs = dbpedia[0].to_list()
        docs = [doc.lower().replace("\n", " ") for doc in docs]
        return docs, None

    def _yahoo_answer(self) -> Tuple[List[str], List[str]]:
        dbpedia = pd.read_csv('/mnt/raid1/gzhou/datasets/yahoo_answer/train.csv', header=None)
        docs = dbpedia[0].to_list()
        docs = [doc.lower().replace("\n", " ") for doc in docs if type(doc) == str]
        return docs, None
        
    def _save(self, docs: List[str], save: str):
        """Save the documents"""
        with open(save, mode="wt", encoding="utf-8") as myfile:
            myfile.write("\n".join(docs))

        self.doc_path = save

    ###################################
    #### sentence level preprocess ####
    ###################################

    # lowercase + base filter
    # some basic normalization
    def f_base(self, s):
        """
        :param s: string to be processed
        :return: processed string: see comments in the source code for more info
        """
        # normalization 1: xxxThis is a --> xxx. This is a (missing delimiter)
        s = re.sub(r'([a-z])([A-Z])', r'\1\. \2', s)  # before lower case
        # normalization 2: lower case
        s = s.lower()
        # normalization 3: "&gt", "&lt"
        s = re.sub(r'&gt|&lt', ' ', s)
        # normalization 4: letter repetition (if more than 2)
        s = re.sub(r'([a-z])\1{2,}', r'\1', s)
        # normalization 5: non-word repetition (if more than 1)
        s = re.sub(r'([\W+])\1{1,}', r'\1', s)
        # normalization 6: string * as delimiter
        s = re.sub(r'\*|\W\*|\*\W', '. ', s)
        # normalization 7: stuff in parenthesis, assumed to be less informal
        s = re.sub(r'\(.*?\)', '. ', s)
        # normalization 8: xxx[?!]. -- > xxx.
        s = re.sub(r'\W+?\.', '.', s)
        # normalization 9: [.?!] --> [.?!] xxx
        s = re.sub(r'(\.|\?|!)(\w)', r'\1 \2', s)
        # normalization 10: ' ing ', noise text
        s = re.sub(r' ing ', ' ', s)
        # normalization 11: noise text
        s = re.sub(r'product received for free[.| ]', ' ', s)
        # normalization 12: phrase repetition
        s = re.sub(r'(.{2,}?)\1{1,}', r'\1', s)
        # Remove Emails
        s = re.sub('\S*@\S*\s?', '', s)
        # Remove new line characters
        s = re.sub('\s+', ' ', s)
        # Remove distracting single quotes(r'([a-z])\1{2,}', r'\1', s)
        # norma
        s = re.sub("\'"," ", s)
        s = re.sub("[^a-zA-Z]+"," ", s)
        return s.strip()


    # language detection
    def f_lan(self, s):
        """
        :param s: string to be processed
        :return: boolean (s is English)
        """

        # some reviews are actually english but biased toward french
        # return detect_language(s) in {'English', 'French','Spanish','Chinese'}
        return detect_language(s) in {'English'}


    ###############################
    #### word level preprocess ####
    ###############################

    # filtering out punctuations and numbers
    def f_punct(self, w_list):
        """
        :param w_list: word list to be processed
        :return: w_list with punct and number filter out
        """
        return [word for word in w_list if word.isalpha()]


    # selecting nouns
    def f_noun(self, w_list):
        """
        :param w_list: word list to be processed
        :return: w_list with only nouns selected
        """
        return [word for (word, pos) in nltk.pos_tag(w_list) if pos[:2] == 'NN']
    
    # typo correction
    def f_typo(self, w_list):
        """
        :param w_list: word list to be processed
        :return: w_list with typo fixed by symspell. words with no match up will be dropped
        """
        w_list_fixed = []
        for word in w_list:
            suggestions = self.sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=3)
            if suggestions:
                w_list_fixed.append(suggestions[0].term)
            else:
                pass
                # do word segmentation, deprecated for inefficiency
                # w_seg = sym_spell.word_segmentation(phrase=word)
                # w_list_fixed.extend(w_seg.corrected_string.split())
        return w_list_fixed


    # def f_stem(self, w_list):
    #     """
    #     :param w_list: word list to be processed
    #     :return: w_list with stemming
    #     """
    #     # stemming if doing word-wise
    #     p_stemmer = PorterStemmer()
    #     return [p_stemmer.stem(word) for word in w_list]


    def f_stopw(self, w_list):
        """
        filtering out stop words
        """
            # filtering out stop words
        # create English stop words list
        stop_words = (list(
            set(get_stop_words('en'))
            |set(get_stop_words('es'))
            |set(get_stop_words('de'))
            |set(get_stop_words('it'))
            |set(get_stop_words('ca'))
            #|set(get_stop_words('cy'))
            |set(get_stop_words('pt'))
            #|set(get_stop_words('tl'))
            |set(get_stop_words('pl'))
            #|set(get_stop_words('et'))
            |set(get_stop_words('da'))
            |set(get_stop_words('ru'))
            #|set(get_stop_words('so'))
            |set(get_stop_words('sv'))
            |set(get_stop_words('sk'))
            #|set(get_stop_words('cs'))
            |set(get_stop_words('nl'))
            #|set(get_stop_words('sl'))
            #|set(get_stop_words('no'))
            #|set(get_stop_words('zh-cn'))
        ))
        return [word for word in w_list if word not in stop_words]


    def make_bigrams(self, texts):
        
        # Build the bigram and trigram models
        bigram = gensim.models.Phrases(texts, min_count=5, threshold=100) # higher threshold fewer phrases.
        # trigram = gensim.models.Phrases(bigram[texts], threshold=100)  

        # Faster way to get a sentence clubbed as a trigram/bigram
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        # trigram_mod = gensim.models.phrases.Phraser(trigram)
        return [bigram_mod[texts]]


    def f_lemmatization(self, texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = self.nlp(" ".join(sent)) 
            for token in doc:
                if token.pos_ in allowed_postags:
                    texts_out.append(token.lemma_)
        return texts_out


    def preprocess_sent(self, rw):
        """
        Get sentence level preprocessed data from raw review texts
        :param rw: review to be processed
        :return: sentence level pre-processed review
        """
        s = self.f_base(rw)
        if not self.f_lan(s):
            return None
        return s


    def preprocess_word(self, s):
        """
        Get word level preprocessed data from preprocessed sentences
        including: remove punctuation, select noun, fix typo, stem, stop_words
        :param s: sentence to be processed
        :return: word level pre-processed review
        """
        if not s:
            return None
        w_list = word_tokenize(s)
        w_list = self.f_punct(w_list)
        w_list = self.f_noun(w_list)
        w_list = self.f_typo(w_list)
        # w_list = self.f_stem(w_list)
        w_list = self.f_stopw(w_list)
        # Do lemmatization keeping only noun, adj, vb, adv
        w_list = self.make_bigrams(w_list)
        w_list = self.f_lemmatization(w_list, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

        return w_list

    def preprocess(self, docs, samp_size=None):
        """
        Preprocess the data
        """
        if not samp_size:
            samp = range(len(docs))
        else:
            samp = np.random.choice(n_docs, samp_size)

        print('Preprocessing raw texts ...')
        n_docs = len(docs)
        sentences = []  # sentence level preprocessed
        token_lists = []  # word level preprocessed
        idx_in = []  # index of sample selected
        
        for i, idx in enumerate(samp):
            sentence = self.preprocess_sent(docs[idx])
            token_list = self.preprocess_word(sentence)
            # token_list = self.preprocess_sent(docs[idx])
            if token_list:
                idx_in.append(idx)
                sentences.append(sentence)
                token_lists.append(token_list)
            print('{} %'.format(str(np.round((i + 1) / len(samp) * 100, 2))), end='\r')
        print('Preprocessing raw texts. Done!')
        return sentences, token_lists, idx_in

    def filter_words(self, docs):
        # vectorizer = TfidfVectorizer(max_df=0.9, min_df=0.006, vocabulary=None, token_pattern=r"(?u)\b[\w|\-]{" + str(3) + r",}\b",  stop_words='english')
        # if self.dataset == "20news":
        # vectorizer = TfidfVectorizer(max_df=0.9, min_df=0.0019, vocabulary=None, token_pattern=r"(?u)\b[\w|\-]{" + str(3) + r",}\b",  stop_words='english')
        # vectorizer = TfidfVectorizer(max_df=0.9, min_df=0.0019, vocabulary=None, token_pattern=r"(?u)\b[\w|\-]{" + str(3) + r",}\b",  stop_words='english')
        # vectorizer = TfidfVectorizer(max_df=0.99, min_df=0.000004, vocabulary=None, token_pattern=r"(?u)\b[\w|\-]{" + str(3) + r",}\b",  stop_words='english')
        # vectorizer = TfidfVectorizer(max_df=0.99, min_df=0.00002, vocabulary=None, token_pattern=r"(?u)\b[\w|\-]{" + str(3) + r",}\b",  stop_words='english')
        # elif self.dataset == "agnews":
        #     vectorizer = TfidfVectorizer(max_df=0.99, min_df=0.00002, vocabulary=None, token_pattern=r"(?u)\b[\w|\-]{" + str(3) + r",}\b",  stop_words='english')
        # elif self.dataset == "yelp":
        vectorizer = TfidfVectorizer(max_df=0.99, min_df=0.00002, vocabulary=None, token_pattern=r"(?u)\b[\w|\-]{" + str(3) + r",}\b",  stop_words='english')
        
        
        vectorizer.fit_transform(docs)
        vocabulary = vectorizer.get_feature_names()
        return vocabulary

    def _preprocess_dataset_custom(self, documents_path, sentence_on=False):
        docs = [line.strip() for line in open(documents_path, 'r').readlines()]
        sentences, token_lists, idx_in = self.preprocess(docs, samp_size=None)
        dsa = [' '.join(word) for word in token_lists]
        vocabulary = self.filter_words(dsa)
        print("created vocab")
        print(len(vocabulary))
        final_docs, final_labels, document_indexes = [], [], []

        for i, doc in enumerate(dsa):
            vocab = set(vocabulary)
            new_doc = [w for w in doc.split() if w in vocab]
            if len(new_doc) > 0:
                final_docs.append(new_doc)
                document_indexes.append(i)

        metadata = {"total_documents": len(dsa), "vocabulary_length": len(vocabulary),
                            "preprocessing-info": 'process'
                            # ,"labels": list(set(final_labels)), "total_labels": len(set(final_labels))
                            }

        if not sentence_on:
            return Dataset(final_docs, vocabulary=vocabulary, metadata=metadata, labels=final_labels,
                                    document_indexes=document_indexes)
        else:
            return Dataset(final_docs, vocabulary=vocabulary, metadata=metadata, labels=final_labels,
                                    document_indexes=document_indexes), sentences
