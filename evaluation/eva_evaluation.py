import argparse
import json
import re
import time
import torch
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm

import sys
import ssl
import yaml
from collections import Counter
from palmettopy.palmetto import Palmetto
from torch.utils.data import DataLoader
import os
from collections import defaultdict

from sklearn.feature_extraction.text import CountVectorizer
from typing import Mapping, Any, List, Tuple

try:
    from bertopic import BERTopic
except ImportError:
    pass

try:
    from top2vec import Top2Vec
except ImportError:
    pass

import gensim
import gensim.corpora as corpora
from gensim.models import ldaseqmodel

import nltk

from contextualized_topic_models.models.ctm import CombinedTM
from contextualized_topic_models.utils.data_preparation import (
    TopicModelDataPreparation,
)


# nltk.download(download_dir="/mnt/raid1/gzhou", info_or_id="stopwords")
# nltk.download("stopwords", download_dir='/mnt/raid1/gzhou/anaconda3/envs/tm/nltk_data')

from nltk.corpus import stopwords

from octis.models.ETM import ETM
from octis.models.LDA import LDA
from octis.models.NMF import NMF
from octis.models.CTM import CTM
from octis.dataset.dataset import Dataset
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.coherence_metrics import Coherence


### TSCTM dependency
from utils import data_utils
from utils.Data import TextData
from runners.Runner import Runner


class Trainer:
    """Train and evaluate a topic model

    Arguments:
        dataset: The dataset to be used, should be a string and either a
                 dataset found in OCTIS or a custom dataset
        model_name: The name of the topic model to be used:
                        * BERTopic
                        * Top2Vec
                        * CTM_CUSTOM (original package)
                        * ETM (OCTIS)
                        * LDA (OCTIS)
                        * CTM (OCTIS)
                        * NMF (OCTIS)
        params: The parameters of the model to be used
        topk: The top n words in each topic to include
        custom_dataset: Whether a custom dataset is used
        bt_embeddings: Pre-trained embeddings used in BERTopic to speed
                       up training.
        bt_timestamps: Timestamps used in BERTopic for dynamic
                       topic modeling
        bt_nr_bins: Number of bins to create from timestamps in BERTopic
        custom_model: A custom BERTopic or Top2Vec class
        verbose: Control the verbosity of the trainer

    Usage:

    ```python
    from evaluation import Trainer
    dataset, custom = "20NewsGroup", False
    params = {"num_topics": [(i+1)*10 for i in range(5)], "random_state": 42}

    trainer = Trainer(dataset=dataset,
                      model_name="LDA",
                      params=params,
                      custom_dataset=custom,
                      verbose=True)
    ```

    Note that we need to specify whether a custom OCTIS dataset is used.
    Since we use a preprocessed dataset from OCTIS [here](https://github.com/MIND-Lab/OCTIS#available-datasets),
    no custom dataset is used.

    This trainer focused on iterating over all combinations of parameters in `params`.
    In the example above, we iterate over different number of topics.
    """

    def __init__(
        self,
        dataset: str,
        model_name: str,
        params: Mapping[str, Any],
        topk: int = 10,
        custom_dataset: bool = False,
        bt_embeddings: np.ndarray = None,
        bt_timestamps: List[str] = None,
        bt_nr_bins: int = None,
        custom_model=None,
        verbose: bool = True,
    ):
        self.dataset = dataset
        self.custom_dataset = custom_dataset
        self.model_name = model_name
        self.params = params
        self.topk = topk
        self.timestamps = bt_timestamps
        self.nr_bins = bt_nr_bins
        self.embeddings = bt_embeddings
        self.ctm_preprocessed_docs = None
        self.custom_model = custom_model
        self.verbose = verbose

        # Prepare data and metrics
        self.data = self.get_dataset()
        self.metrics = self.get_metrics()

        # CTM
        self.qt_ctm = None
        self.training_dataset_ctm = None

    def train(self, save: str = False) -> Mapping[str, Any]:
        """Train a topic model

        Arguments:
            save: The name of the file to save it to.
                  It will be saved as a .json in the current
                  working directory

        Usage:

        ```python
        from evaluation import Trainer
        dataset, custom = "20NewsGroup", False
        params = {"num_topics": [(i+1)*10 for i in range(5)], "random_state": 42}

        trainer = Trainer(dataset=dataset,
                        model_name="LDA",
                        params=params,
                        custom_dataset=custom,
                        verbose=True)
        results = trainer.train(save="LDA_results")
        ```
        """

        results = []

        # Loop over all parameters
        params_name = list(self.params.keys())
        params = {
            param: (value if type(value) == list else [value])
            for param, value in self.params.items()
        }
        new_params = list(itertools.product(*params.values()))

        print("First 5 mode")
        mean_coherence_ca, mean_coherence_cp, mean_coherence_npmi = [], [], []
        for param_combo in new_params:
            # Train and evaluate model
            params_to_use = {
                param: value for param, value in zip(params_name, param_combo)
            }
            output, computation_time = self._train_tm_model(params_to_use)
            print(output, computation_time)

            topnum = 0
            if self.model_name == "malletLDA":
                topnum = params_to_use["num_topics"]
            elif self.model_name == "BERTopic" or self.model_name == "Top2Vec":
                topnum = params_to_use["nr_topics"]
            elif self.model_name == "CTM_CUSTOM":
                topnum = params_to_use["n_components"]
            elif self.model_name == "tsctm":
                topnum = params_to_use["num_topics"]
            elif self.model_name == "rlmodel":
                topnum = params_to_use["num_topics"]
            coherence_ca, coherence_cp = [], []
            print(f"Topic Number is {topnum}")
            all_topics = []
            for tops in output["topics"]:
                all_topics.extend(tops)
            whole_topics = Counter(all_topics)
            final_candidates = []
            palmetto = Palmetto(
                palmetto_uri="http://localhost:7777/service/",
                timeout=1000,
            )
            for tops in tqdm(output["topics"]):
                temp = []
                for item in tops:
                    temp.append(1 / whole_topics[item])
                final_candidates.append(1 / len(tops) * sum(temp))
                coherence_cp.append(palmetto.get_coherence(tops, coherence_type="cp"))
                coherence_ca.append(palmetto.get_coherence(tops, coherence_type="ca"))
            scores = self.evaluate(output)
            # scores.update({"Topic Uniqueness": np.mean(final_candidates)})
            mean_ca, mean_cp = np.mean(coherence_ca), np.mean(coherence_cp)
            scores.update({"C_A": mean_ca})
            scores.update({"C_P": mean_cp})
            print(f"Score origin from OCTIS are: {scores}")

            # Update results
            if self.model_name == "Top2Vec" or self.model_name == "BERTopic":
                print(f"Using model: {self.model_name}")
                result = {
                    "Dataset": self.dataset,
                    "Dataset Size": len(self.data.get_corpus()),
                    "Model": self.model_name,
                    "Topic Numbers": params_to_use["nr_topics"],
                    "Computation Time": computation_time,
                    "Score Summary": scores,
                    # "Coherence C_A": np.mean(coherence_ca),
                    # "Coherence C_P": np.mean(coherence_cp),
                    # "Coherence C_NPMI": np.mean(coherence_npmi),
                }
            elif self.model_name == "BERTopic":
                print(f"Using model: {self.model_name}")
                result = {
                    "Dataset": self.dataset,
                    "Dataset Size": len(self.data.get_corpus()),
                    "Model": self.model_name,
                    "Topic Numbers": params_to_use["nr_topics"],
                    "Computation Time": computation_time,
                    "Score Summary": scores,
                }
            elif self.model_name == "malletLDA":
                print(f"Using model: {self.model_name}")
                result = {
                    "Dataset": self.dataset,
                    "Dataset Size": len(self.data.get_corpus()),
                    "Model": self.model_name,
                    "Topic Numbers": params_to_use["num_topics"],
                    "Computation Time": computation_time,
                    "Score Summary": scores,
                }
            if self.model_name == "CTM_CUSTOM":
                print(f"Using model: {self.model_name}")
                result = {
                    "Dataset": self.dataset,
                    "Dataset Size": len(self.data.get_corpus()),
                    "Model": self.model_name,
                    "Topic Numbers": params_to_use["n_components"],
                    "Computation Time": computation_time,
                    "Score Summary": scores,
                }
            if self.model_name == "tsctm":
                print(f"Using model: {self.model_name}")
                result = {
                    "Dataset": self.dataset,
                    "Dataset Size": len(self.data.get_corpus()),
                    "Model": self.model_name,
                    "Topic Numbers": params_to_use["num_topics"],
                    "Computation Time": computation_time,
                    "Score Summary": scores,
                }
            if self.model_name == "rlmodel":
                print(f"Using model: {self.model_name}")
                result = {
                    "Dataset": self.dataset,
                    "Dataset Size": len(self.data.get_corpus()),
                    "Model": self.model_name,
                    "Topic Numbers": params_to_use["num_topics"],
                    "Computation Time": computation_time,
                    "Score Summary": scores,
                }
            results.append(result)

            if save:
                with open(f"{save}.json", "w") as f:
                    json.dump(results, f)

        return results

    def _train_tm_model(
        self, params: Mapping[str, Any]
    ) -> Tuple[Mapping[str, Any], float]:
        """Select and train the Topic Model"""
        # Train custom CTM
        if self.model_name == "CTM_CUSTOM":
            if self.qt_ctm is None:
                self._preprocess_ctm()
            return self._train_ctm(params)

        # Train BERTopic
        elif self.model_name == "BERTopic":
            print(f"Running the model: {self.model_name}")
            return self._train_bertopic(params)

        # Train Top2Vec
        elif self.model_name == "Top2Vec":
            print(f"Running the model: {self.model_name}")
            return self._train_top2vec(params)

        # Train LDAseq
        elif self.model_name == "LDAseq":
            return self._train_ldaseq(params)

        # Train malletLDA
        elif self.model_name == "malletLDA":
            print(f"Running the model: {self.model_name}")
            return self._train_malletlda(params)

        elif self.model_name == "tsctm":
            print(f"Running the model: {self.model_name}")
            return self._train_tsctm(params)
        
        elif self.model_name == "rlmodel":
            print(f"Running the model: {self.model_name}")
            return self._train_rl_tm(params)

        # Train OCTIS model
        octis_models = ["ETM", "LDA", "CTM", "NMF"]
        if self.model_name in octis_models:
            print(f"Running the model: {self.model_name}")
            return self._train_octis_model(params)

    def _train_malletlda(
        self, params: Mapping[str, any]
    ) -> Tuple[Mapping[str, Any], float]:
        """Train mallet LDA model"""
        data = self.data.get_corpus()
        docs = [" ".join(words) for words in data]

        data_words = list(sent_to_words(docs))
        id2word = corpora.Dictionary(data_words)
        corpus = [id2word.doc2bow(text) for text in data_words]

        params["corpus"] = corpus
        params["id2word"] = id2word
        params[
            "mallet_path"
        ] = "/mnt/raid1/gzhou/mallet-2.0.8/bin/mallet"  # update this path

        start = time.time()
        ldamallet = gensim.models.wrappers.LdaMallet(**params)
        end = time.time()
        computation_time = end - start

        tops = ldamallet.print_topics()
        tops = [" ".join(doc[1].split("+")) for doc in tops]
        tops = [re.sub("[^a-zA-Z]+", " ", sent) for sent in tops]
        tops = [x.strip() for x in tops if x.strip() != ""]
        tops = [sent.split(" ") for sent in tops]

        output_tm = {"topics": tops}
        return output_tm, computation_time

    def _train_ldaseq(
        self, params: Mapping[str, any]
    ) -> Tuple[Mapping[str, Any], float]:
        """Train LDA seq model"""
        data = self.data.get_corpus()
        docs = [" ".join(words) for words in data]

        df = pd.DataFrame({"Doc": docs, "Timestamp": self.timestamps}).sort_values(
            "Timestamp"
        )
        df["Bins"] = pd.cut(df.Timestamp, bins=params["nr_bins"])
        df["Timestamp"] = df.apply(lambda row: row.Bins.left, 1)
        timestamps = df.groupby("Bins").count().Timestamp.values
        docs = df.Doc.values

        data_words = list(sent_to_words(docs))
        id2word = corpora.Dictionary(data_words)
        corpus = [id2word.doc2bow(text) for text in data_words]

        print(len(corpus), len(self.timestamps), timestamps)

        params["corpus"] = corpus
        params["id2word"] = id2word
        params["time_slice"] = timestamps
        del params["nr_bins"]

        start = time.time()
        ldaseq = ldaseqmodel.LdaSeqModel(**params)
        end = time.time()
        computation_time = end - start

        all_topics = {}
        for i in range(len(timestamps)):
            topics = ldaseq.print_topics(time=i)
            topics = [[word for word, _ in topic][:5] for topic in topics]
            all_topics[i] = {"topics": topics}

        return all_topics, computation_time

    def _train_top2vec(
        self, params: Mapping[str, Any]
    ) -> Tuple[Mapping[str, Any], float]:
        """Train Top2Vec"""
        nr_topics = None
        data = self.data.get_corpus()
        data = [" ".join(words) for words in data]
        params["documents"] = data

        if params.get("nr_topics"):
            nr_topics = params["nr_topics"]
            del params["nr_topics"]

        start = time.time()

        if self.custom_model is not None:
            model = self.custom_model(**params)
        else:
            model = Top2Vec(**params)

        if nr_topics:
            try:
                _ = model.hierarchical_topic_reduction(nr_topics)
                params["reduction"] = True
                params["nr_topics"] = nr_topics
            except:
                params["reduction"] = False
                nr_topics = False

        end = time.time()
        computation_time = float(end - start)

        if nr_topics:
            topic_words, _, _ = model.get_topics(reduced=True)
        else:
            topic_words, _, _ = model.get_topics(reduced=False)

        topics_old = [list(topic[:10]) for topic in topic_words]
        all_words = [word for words in self.data.get_corpus() for word in words]
        topics = []
        for topic in topics_old:
            words = []
            for word in topic:
                if word in all_words:
                    words.append(word)
                else:
                    print(f"error: {word}")
                    words.append(all_words[0])
            topics.append(words)

        if not nr_topics:
            params["nr_topics"] = len(topics)
            params["reduction"] = False

        del params["documents"]
        output_tm = {
            "topics": topics,
        }
        return output_tm, computation_time

    def _train_ctm(self, params) -> Tuple[Mapping[str, Any], float]:
        """Train CTM"""
        params["bow_size"] = len(self.qt_ctm.vocab)
        ctm = CombinedTM(**params)

        start = time.time()
        ctm.fit(self.training_dataset_ctm)
        end = time.time()
        computation_time = float(end - start)

        topics = ctm.get_topics(10)
        topics = [topics[x] for x in topics]

        output_tm = {
            "topics": topics,
        }

        return output_tm, computation_time

    def _train_tsctm(
        self, params: Mapping[str, any]
    ) -> Tuple[Mapping[str, Any], float]:
        """Train mallet TSCTM model"""
        data = self.data.get_corpus()
        docs = [" ".join(words) for words in data]
        print(f'Running dataset: {params["dataset_name"]}, number of topic: {params["num_topics"]}')

        parser = argparse.ArgumentParser()
        config = parser.parse_args()
        config = vars(config)
        config.update(
            {
                "model": "TSCTM",
                "num_epoch": 200,
                "batch_size": 200,
                "learning_rate": 0.002,
                "num_topic": params["num_topics"],
                "en1_units": 100,
                "num_top_word": 15,
                "test_index": 1,
            }
        )
        config = argparse.Namespace(**config)

        data_utils.update_args(
            config,
            "/home/joe/projects/tm-projects/topic-modeling/plugins/TSCTM/configs/TSCTM.yaml",
        )
        print("===>Info: args: \n", yaml.dump(vars(config), default_flow_style=False))

        device = "cuda" if torch.cuda.is_available() else "cpu"

        train_dataset = TextData(docs, device)
        train_loader = DataLoader(train_dataset, config.batch_size, shuffle=True)
        config.vocab_size = len(train_dataset.vocab)

        # Training
        model_runner = Runner(config, device)
        start = time.time()
        beta = model_runner.train(train_loader)
        end = time.time()
        computation_time = float(end - start)

        topic_str_list = data_utils.print_topic_words(
            beta, train_dataset.vocab, config.num_top_word
        )

        topics = [sent.split() for sent in topic_str_list]
        print(topics)
        print(len(topics))
        output_tm = {
            "topics": topics,
        }

        return output_tm, computation_time
    
    def _train_rl_tm(
        self, params: Mapping[str, any]
    ) -> Tuple[Mapping[str, Any], float]:
        """Train mallet TSCTM model"""
        start = time.time()
        
        print(f'Running dataset: {params["dataset_name"]}, number of topic: {params["num_topics"]}')

        folder_path = '/home/joe/projects/tm-projects/rl-for-topic-models/evals/figures'
        topic_dict = defaultdict(lambda: defaultdict(list))

        if os.path.exists(folder_path):
            filenames = os.listdir(folder_path)
            for filename in filenames:
                if filename.endswith('.txt'):
                    dataset_name = filename.split('_')[-2]
                    topic_number = filename.split('_')[-1].split('.')[-2].lstrip('topk')
                    if dataset_name == "answer":
                        dataset_name = "yahoo_answer"
                    elif dataset_name == "20newgroups":
                        dataset_name = "20NewsGroup"

                    full_path = os.path.join(folder_path, filename)
                    file_content = []
                    
                    with open(full_path, 'r') as file:
                        for line in file:
                            line = line.strip()
                            file_content.append(line.split(' '))
                    topic_dict[dataset_name][topic_number] = file_content

        topics = topic_dict[params["dataset_name"]][str(params["num_topics"])]

        output_tm = {
            "topics": [a[:15] for a in topics],
        }
        end = time.time()
        computation_time = float(end - start)
        return output_tm, computation_time

    def _preprocess_ctm(self):
        """Preprocess data for CTM"""
        # Prepare docs
        data = self.data.get_corpus()
        docs = [" ".join(words) for words in data]

        # Remove stop words
        stop_words = stopwords.words("english")
        preprocessed_documents = [
            " ".join([x for x in doc.split(" ") if x not in stop_words]).strip()
            for doc in docs
        ]

        # Get vocabulary
        vectorizer = CountVectorizer(
            max_features=2000, token_pattern=r"\b[a-zA-Z]{2,}\b"
        )
        vectorizer.fit_transform(preprocessed_documents)
        vocabulary = set(vectorizer.get_feature_names())

        # Preprocess documents further
        preprocessed_documents = [
            " ".join([w for w in doc.split() if w in vocabulary]).strip()
            for doc in preprocessed_documents
        ]

        # Prepare CTM data
        qt = TopicModelDataPreparation("all-mpnet-base-v2")
        training_dataset = qt.fit(
            text_for_contextual=docs, text_for_bow=preprocessed_documents
        )

        self.qt_ctm = qt
        self.training_dataset_ctm = training_dataset

    def _train_octis_model(
        self, params: Mapping[str, any]
    ) -> Tuple[Mapping[str, Any], float]:
        """Train OCTIS model"""

        if self.model_name == "ETM":
            model = ETM(**params)
            model.use_partitions = False
        elif self.model_name == "LDA":
            model = LDA(**params)
            model.use_partitions = False
        elif self.model_name == "CTM":
            model = CTM(**params)
            model.use_partitions = False
        elif self.model_name == "NMF":
            model = NMF(**params)
            model.use_partitions = False

        start = time.time()
        output_tm = model.train_model(self.data)
        end = time.time()
        computation_time = end - start
        return output_tm, computation_time

    def _train_bertopic(
        self, params: Mapping[str, any]
    ) -> Tuple[Mapping[str, Any], float]:
        """Train BERTopic model"""
        data = self.data.get_corpus()
        data = [" ".join(words) for words in data]
        params["calculate_probabilities"] = False

        if self.custom_model is not None:
            model = self.custom_model(**params)
        else:
            model = BERTopic(**params)

        start = time.time()
        topics, _ = model.fit_transform(data, self.embeddings)

        # Dynamic Topic Modeling
        if self.timestamps:
            topics_over_time = model.topics_over_time(
                data,
                topics,
                self.timestamps,
                nr_bins=self.nr_bins,
                evolution_tuning=False,
                global_tuning=False,
            )
            unique_timestamps = topics_over_time.Timestamp.unique()
            dtm_topics = {}
            for unique_timestamp in unique_timestamps:
                dtm_topic = topics_over_time.loc[
                    topics_over_time.Timestamp == unique_timestamp, :
                ].sort_values("Frequency", ascending=True)
                dtm_topic = dtm_topic.loc[dtm_topic.Topic != -1, :]
                dtm_topic = [topic.split(", ") for topic in dtm_topic.Words.values]
                dtm_topics[unique_timestamp] = {"topics": dtm_topic}

                all_words = [word for words in self.data.get_corpus() for word in words]

                updated_topics = []
                for topic in dtm_topic:
                    updated_topic = []
                    for word in topic:
                        if word not in all_words:
                            print(word)
                            updated_topic.append(all_words[0])
                        else:
                            updated_topic.append(word)
                    updated_topics.append(updated_topic)

                dtm_topics[unique_timestamp] = {"topics": updated_topics}

            output_tm = dtm_topics

        end = time.time()
        computation_time = float(end - start)

        if not self.timestamps:
            all_words = [word for words in self.data.get_corpus() for word in words]
            bertopic_topics = [
                [
                    vals[0] if vals[0] in all_words else all_words[0]
                    for vals in model.get_topic(i)[:10]
                ]
                for i in range(len(set(topics)) - 1)
            ]

            output_tm = {"topics": bertopic_topics}

        return output_tm, computation_time

    def evaluate(self, output_tm):
        """Using metrics and output of the topic model, evaluate the topic model"""
        if self.timestamps:
            results = {str(timestamp): {} for timestamp, _ in output_tm.items()}
            for timestamp, topics in output_tm.items():
                self.metrics = self.get_metrics()
                for scorers, _ in self.metrics:
                    for scorer, name in scorers:
                        score = scorer.score(topics)
                        results[str(timestamp)][name] = float(score)

        else:
            # Calculate results
            results = {}
            for scorers, _ in self.metrics:
                for scorer, name in scorers:
                    score = scorer.score(output_tm)
                    results[name] = float(score)

            # Print results
            if self.verbose:
                print("Results")
                print("============")
                for metric, score in results.items():
                    print(f"{metric}: {str(score)}")

        return results

    def get_dataset(self):
        """Get dataset from OCTIS"""
        data = Dataset()

        if self.custom_dataset:
            data.load_custom_dataset_from_folder(self.dataset)
        else:
            data.fetch_dataset(self.dataset)
        return data

    def get_metrics(self):
        """Prepare evaluation measures using OCTIS"""
        cv = Coherence(texts=self.data.get_corpus(), topk=self.topk, measure="c_v")
        uci = Coherence(texts=self.data.get_corpus(), topk=self.topk, measure="c_uci")
        npmi = Coherence(texts=self.data.get_corpus(), topk=self.topk, measure="c_npmi")
        topic_diversity = TopicDiversity(topk=self.topk)

        # Define methods
        coherence_cv = [(cv, "cv")]
        coherence_npmi = [(npmi, "npmi")]
        coherence_uci = [(uci, "uci")]
        diversity = [(topic_diversity, "diversity")]
        metrics = [
            (coherence_cv, "Coherence_cv"),
            (coherence_npmi, "Coherence_npmi"),
            (coherence_uci, "Coherence_uci"),
            (diversity, "Topic Diversity"),
        ]

        return metrics
        # """Prepare evaluation measures using OCTIS"""
        # npmi = Coherence(texts=self.data.get_corpus(), topk=self.topk, measure="c_npmi")
        # topic_diversity = TopicDiversity(topk=self.topk)

        # # Define methods
        # coherence = [(npmi, "npmi")]
        # diversity = [(topic_diversity, "diversity")]
        # metrics = [(coherence, "Coherence"), (diversity, "Diversity")]

        # return metrics


def sent_to_words(sentences):
    for sentence in sentences:
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))
