###############################
# imports
###############################

from time import time
from datetime import timedelta
import os
import random
import json
import gc

import pandas as pd
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub

from sklearn.cluster import MiniBatchKMeans, KMeans, AffinityPropagation, DBSCAN
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.datasets.samples_generator import make_blobs
from sklearn import metrics
import distance
from scipy.spatial import cKDTree

import re
from collections import Counter
from itertools import combinations
from random import randint
from sklearn.externals import joblib

from . import model
from . import sample
from . import encoder


class ArticleGenerator:

    def __init__(self):
        ###############################
        # configuration
        ###############################

        self.default_path = ""

        self.module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]

        # PARSE_TXT_FILES = False #@param ["True", "False"] {type:"raw"}
        # LOAD_PROCESSED_SENTENCES = True #@param ["True", "False"] {type:"raw"}
        # 0 - no data
        # 1 - TXT files already parsed to data_df
        # 2 - questions already processed and stored to clustered_questions_df
        # 3 - sentences already extracted and stored to meta_with_texts.csv and all_sentences.json
        # 4 - sentences already sub-clusterized and stored to clustered_sentences
        # 5 - structure of texts is extracted and stored to data_df (meta_with_texts.csv)
        # 6 - data, articles and model ready
        self.STAGE = 1 #@param [0, 1, 2, 3, 4, 5, 6] {type:"raw"}
        self.CLUSTER_ALGORITHM = "DBSCAN" #@param ["KMeans", "AffProp", "DBSCAN"]

        self.AVG_EXPECTED_QUESTIONS_PER_CLUSTER = 50
        self.AVG_EXPECTED_SENTENCES_PER_CLUSTER = 5

        self.BATCH_SIZE = 1000

        self.data_df = None
        self.questions = None
        self.embed_module = None
        self.clustered_questions_df = None
        self.START_STEP = 0
        self.gpt_questions_df = None
        self.all_sentences = None

    ###############################
    # STEP 1: load main dataframe, extract questions
    ###############################
    def step_load_data(self):
        if self.STAGE < 1:
            self.data_df = pd.read_csv(self.default_path + 'data/meta.csv')
            self.data_df['text'] = ""
        else:
            self.data_df = pd.read_csv(self.default_path + 'data/meta_with_texts.csv')

        self.questions = self.data_df['question']
        return self.data_df

    ###############################
    # STEP 2: load and init Universal Sentence Encoder's TF Hub module
    # Universal Sentence Encoder used for convert sentences to 512 dimensional float (-1, 1) vectors
    ###############################
    def step_prepare_tf_hub(self):
        # Import the Universal Sentence Encoder's TF Hub module
        self.embed_module = hub.Module(self.module_url)

        # Reduce logging output.
        tf.logging.set_verbosity(tf.logging.ERROR)

        return self.embed_module

    ###############################
    # STEP 3: find clusters among questions
    # Found clusters stores to clustered_questions_df.
    # Then field cluster_id from clustered_questions_df connect to field question_cluster_id from main dataframe.
    ###############################
    def step_clusterize_questions(self):
        num_clusters = int(len(self.questions) / self.AVG_EXPECTED_QUESTIONS_PER_CLUSTER)

        def fill_by_questions(q):
            i = np.where(unique_questions == q)
            return questions_clusters[i[0][0]]

        if self.STAGE < 2:

            unique_questions = np.unique(self.questions)

            questions_clusters, center_indexes = self.clusterize_it(unique_questions,
                                                                    self.CLUSTER_ALGORITHM,
                                                                    num_clusters,
                                                                    verbose=1)

            s1 = pd.Series(self.questions, name='question')
            s2 = pd.Series(s1.apply(fill_by_questions), name='cluster_id')

            self.clustered_questions_df = pd.concat([self.data_df['id'], s1, s2], axis=1)
            self.clustered_questions_df.to_csv(self.default_path + 'data/clustered_questions_df.csv')

        else:
            self.clustered_questions_df = pd.read_csv(self.default_path + 'data/clustered_questions_df.csv')

        self.data_df['question_cluster_id'] = self.clustered_questions_df['cluster_id']
        return self.clustered_questions_df

    ###############################
    # STEP 4: generate texts for question's clusters
    # Here we take any question from cluster, pass it to GPT-2 and save text to gpt_questions_df
    ###############################
    def step_generate_questions_texts(self, START_STEP):
        question_clusters_ids = self.clustered_questions_df['cluster_id']
        question_clusters_ids = np.unique(question_clusters_ids)
        print('question_clusters_ids', len(question_clusters_ids))
        typical_questions = []

        for cluster_id in question_clusters_ids:
            typical_question = self.clustered_questions_df[self.clustered_questions_df['cluster_id'] == cluster_id]['question'].values
            # non-clustered
            if cluster_id == -1:
                for q in typical_question:
                    typical_questions.append(q)
            else:
                if len(typical_question) > 0:
                    typical_questions.append(typical_question[0])

        print("num typical_questions:", len(typical_questions))
        typical_questions = np.unique(typical_questions)

        print("num typical_questions:", len(typical_questions))

        if START_STEP == 0:
            texts = [''] * len(typical_questions)
            s1 = pd.Series(typical_questions, name='question')
            s2 = pd.Series(texts, name='text')
            self.gpt_questions_df = pd.concat([s1, s2], axis=1)
        else:
            self.gpt_questions_df = pd.read_csv(self.default_path + 'data/gpt_questions_df.csv')

        if START_STEP == len(question_clusters_ids):
            texts = self.interact_model(typical_questions, self.gpt_questions_df, START_STEP,
                                        model_name='345M', seed=77, top_k=40, verbose=1)

        return self.gpt_questions_df

    ###############################
    # STEP 5: fill question's clusters with generated texts
    # Save it to clustered_questions_df
    ###############################
    def step_merge_texts_to_questions(self):

        clustered_questions_df = self.clustered_questions_df
        gpt_questions_df = self.gpt_questions_df

        def fill_by_cluster_ids(q):
            cluster_id = clustered_questions_df[clustered_questions_df['question'] == q]['cluster_id'].values

            if len(cluster_id) > 0:
                return cluster_id[0]

            return -2

        self.gpt_questions_df['cluster_id'] = self.gpt_questions_df['question'].apply(fill_by_cluster_ids)

        #

        def fill_by_texts(question):
            text = gpt_questions_df[gpt_questions_df['question'] == question]['text'].values

            if len(text) > 0:
                return text[0]

            cluster_id = clustered_questions_df[clustered_questions_df['question'] == question]['cluster_id'].values

            if len(cluster_id) > 0:
                text = gpt_questions_df[gpt_questions_df['cluster_id'] == cluster_id[0]]['text'].values

                if len(text) > 0:
                    return text[0]

            return ''

        self.clustered_questions_df['text'] = self.clustered_questions_df['question'].apply(fill_by_texts)

    ###############################
    #  step
    ###############################
    def step_load_texts(self):
        if self.STAGE < 1:
            self.fill_df_with_texts(self.data_df)

    ###############################
    #  step
    ###############################
    def step_extract_sentences(self):
        if self.STAGE < 3:
            self.all_sentences = []

            start_time = time()

            # num_clusters = int(len(self.questions) / self.AVG_EXPECTED_QUESTIONS_PER_CLUSTER)
            num_clusters = np.max(self.clustered_questions_df['cluster_id']) + 1

            text_ids = []
            group_obj = self.clustered_questions_df.groupby('cluster_id')
            for i in range(num_clusters):
                if i in group_obj.groups.keys():
                    text_ids.append(group_obj.get_group(i)['id'].values)
                else:
                    text_ids.append([])

            self.data_df['split_sentences'] = ""
            for i in range(num_clusters):
                sentences = self.extract_sentences(self.data_df, text_ids[i])
                sentences = np.unique(sentences)
                self.all_sentences.append(sentences.tolist())

                if i % 100 == 0:
                    elapsed_time = time() - start_time
                    print("{}/{} elapsed time: {}".format(i + 1, num_clusters, timedelta(seconds=elapsed_time)))

            elapsed_time = time() - start_time
            print("{}/{} total elapsed time: {}".format(i + 1, num_clusters, timedelta(seconds=elapsed_time)))

            print('save all_sentences.npy')
            with open(self.default_path + 'data/all_sentences.json', 'w') as outfile:
                json.dump(self.all_sentences, outfile)
            #     np.save(default_path + 'data/all_sentences.npy', all_sentences)

            self.data_df.to_csv(self.default_path + 'data/meta_with_texts.csv')

        else:
            with open(self.default_path + 'data/all_sentences.json') as infile:
                self.all_sentences = json.load(infile)

    ###############################
    #  step
    ###############################
    def step_find_closest_sentences_to_question(self):
        # self.clustered_questions_df['closest_sentences'] = ""
        start_all_time = time()
        num_clusters = len(self.all_sentences)
        total_all_sentences = []
        total_all_indexes = []
        total_all_questions = []
        questions_indexes = []
        last_index = 0
        for cluster_id in range(num_clusters):
            total_all_sentences.extend(self.all_sentences[cluster_id])
            total_all_indexes.append([x for x in range(last_index, last_index + len(self.all_sentences[cluster_id]))])
            last_index += len(self.all_sentences[cluster_id])

            question = self.clustered_questions_df[self.clustered_questions_df['cluster_id'] == cluster_id]['question'].values
            if len(question) > 0:
                total_all_questions.append(question[0])
                questions_indexes.append(len(total_all_questions) - 1)
            else:
                questions_indexes.append(None)

        print("Num all sentences:", len(total_all_sentences))
        start_time = time()
        total_all_sentences_embeddings = self.get_embedings(total_all_sentences, verbose=0)
        questions_embeddings = self.get_embedings(total_all_questions, verbose=0)
        elapsed_time = time() - start_time
        print("elapsed time: {}".format(timedelta(seconds=elapsed_time)))

        for cluster_id in range(num_clusters):
            start_time = time()
            
            sentences_indexes = total_all_indexes[cluster_id]
            sentences = []
            for i in sentences_indexes:
                sentences.append(total_all_sentences_embeddings[i])

            question = None
            if questions_indexes[cluster_id] is not None:
                question = questions_embeddings[questions_indexes[cluster_id]]

            if question is not None and len(sentences) > 0:
                closest = self.find_closest_to(sentences, question, 20)
                sentences = np.array(sentences)
                closest = np.array(closest)
                closest = closest[closest < len(sentences)]
                self.clustered_questions_df.loc[
                    self.clustered_questions_df['cluster_id'] == cluster_id,
                    ['closest_sentences']
                ] = json.dumps(sentences[closest].tolist())

            elapsed_time = time() - start_time
            total_elapsed_time = time() - start_all_time
            print('cluster_id:', cluster_id, "num sentences:", len(sentences))
            print("{}/{} elapsed time: {}, total time: {}".format(cluster_id, num_clusters, timedelta(seconds=elapsed_time), timedelta(seconds=total_elapsed_time)))

            if cluster_id % 10 == 0:
                gc.collect()

    ###############################
    #  step
    ###############################

    ###############################
    ###############################
    ###############################

    ###############################
    # extract embedings
    ###############################

    def get_embedings(self, strings, verbose=0):
        embeddings = np.array([])

        similarity_input_placeholder = tf.placeholder(tf.string, shape=(None))
        similarity_message_encodings = self.embed_module(similarity_input_placeholder)

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            session.run(tf.tables_initializer())

            start_time = time()
            num_steps = int(len(strings) / self.BATCH_SIZE) + 1
            for i in range(num_steps):
                if i * self.BATCH_SIZE < len(strings):
                    embeddings_batch = session.run(
                        similarity_message_encodings, feed_dict={ similarity_input_placeholder: strings[i * self.BATCH_SIZE : (i + 1) * self.BATCH_SIZE] }
                    )

                    if len(embeddings) == 0:
                        embeddings = embeddings_batch
                    else:
                        embeddings = np.vstack((embeddings, embeddings_batch))

                    elapsed_time = time() - start_time
                    if verbose > 0:
                        print ("step", (i+1), "from", num_steps, ", elapsed time", timedelta(seconds=elapsed_time))

        return embeddings

    ###############################
    # clusterization based on their embedings [K-Means]
    ###############################

    def clusterize_it(self, strings, algorithm, num_clusters, verbose=0):

        embeddings = self.get_embedings(strings, verbose=0)

        start_time = time()

        if algorithm == 'KMeans':
            kmeans = KMeans(n_clusters=num_clusters, verbose=0).fit(embeddings)

            clusters = kmeans.predict(embeddings)

            closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, embeddings)

        elif algorithm == 'DBSCAN':

            MAX_CLUSTER_SIZE = 250

            samples = embeddings
            multiplicator = 1
            all_done = False
            final_clusters = [-1] * len(embeddings)
            final_closest = [-1] * len(embeddings)
            last_cluster_id = 0
            while not all_done:
                if verbose > 1:
                    print('multiplicator:', multiplicator)
                clustering = DBSCAN(eps=0.4/multiplicator, min_samples=2).fit(samples)
                clusters = clustering.labels_
                closest = clustering.core_sample_indices_

                masking = clusters >= 0
                clusters = clusters + masking * last_cluster_id
                #             closest = closest + masking * last_cluster_id

                unique, counts = np.unique(clusters, return_counts=True)

                if verbose > 1:
                    print('counts', counts)
                    print('unique', unique)
                #             print('closest', closest)
                #             print('closest', len(closest))

                last_cluster_id = np.max(unique) + 1

                repeat_indicies = []
                all_done = True
                for i in range(len(counts)):
                    if counts[i] > MAX_CLUSTER_SIZE:
                        repeat_indicies.append(unique[i].item())
                        if unique[i] != -1:
                            all_done = False

                if verbose > 1:
                    print('repeat_indicies', repeat_indicies)
                new_samples = []
                for i in repeat_indicies:
                    new_samples.extend(samples[clusters == i].tolist())

                j = 0
                for i in range(len(clusters)):
                    if clusters[i] not in repeat_indicies:
                        final_clusters[j] = clusters[i]
                    #                     final_closest[j] = closest[i]

                    j += 1
                    if j < len(final_clusters):
                        while final_clusters[j] != -1:
                            j += 1
                            if j >= len(final_clusters):
                                break

                samples = np.array(new_samples)

                multiplicator += 0.3

            last_cluster_id = np.max(final_clusters) + 1
            for i in range(len(final_clusters)):
                if final_clusters[i] == -1:
                    final_clusters[i] = last_cluster_id
                    last_cluster_id += 1

        elapsed_time = time() - start_time
        if verbose > 0:
            print("total elapsed time:", timedelta(seconds=elapsed_time))

        return final_clusters, closest

    ###############################
    # generate texts using GPT-2
    ###############################

    def interact_model(
            self,
            strings,
            save_df,
            start_step,
            model_name='117M',
            seed=None,
            nsamples=1,
            batch_size=1,
            length=None,
            temperature=1,
            top_k=0,
            models_dir='models',
            verbose=0
    ):
        """
        Interactively run the model
        :model_name=117M : String, which model to use
        :seed=None : Integer seed for random number generators, fix seed to reproduce
         results
        :nsamples=1 : Number of samples to return total
        :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
        :length=None : Number of tokens in generated text, if None (default), is
         determined by model hyperparameters
        :temperature=1 : Float value controlling randomness in boltzmann
         distribution. Lower temperature results in less random completions. As the
         temperature approaches zero, the model will become deterministic and
         repetitive. Higher temperature results in more random completions.
        :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
         considered for each step (token), resulting in deterministic completions,
         while 40 means 40 words are considered at each step. 0 (default) is a
         special setting meaning no restrictions. 40 generally is a good value.
         :models_dir : path to parent folder containing model subfolders
         (i.e. contains the <model_name> folder)
        """
        models_dir = os.path.expanduser(os.path.expandvars(models_dir))
        if batch_size is None:
            batch_size = 1
        assert nsamples % batch_size == 0

        enc = encoder.get_encoder(model_name, models_dir)
        hparams = model.default_hparams()
        with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
            hparams.override_from_dict(json.load(f))

        if length is None:
            length = hparams.n_ctx // 2
        elif length > hparams.n_ctx:
            raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

        with tf.Session(graph=tf.Graph()) as sess:
            context = tf.placeholder(tf.int32, [batch_size, None])
            np.random.seed(seed)
            tf.set_random_seed(seed)
            output = sample.sample_sequence(
                hparams=hparams, length=length,
                context=context,
                batch_size=batch_size,
                temperature=temperature, top_k=top_k
            )

            saver = tf.train.Saver()
            ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
            saver.restore(sess, ckpt)

            start_time = time()
            num_steps = len(strings)

            texts = save_df['text'].values
            for step in range(start_step, num_steps):
                raw_text = strings[step]
                context_tokens = enc.encode(raw_text)
                generated = 0
                for _ in range(nsamples // batch_size):
                    out = sess.run(output, feed_dict={
                        context: [context_tokens for _ in range(batch_size)]
                    })[:, len(context_tokens):]
                    for i in range(batch_size):
                        generated += 1
                        text = enc.decode(out[i])
                        if verbose > 1:
                            print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                            print(text)

                if verbose > 1:
                    print("=" * 80)

                texts[step] = text

                if step % 100 == 0:

                    elapsed_time = time() - start_time
                    if verbose > 0:
                        print("step", step, "from", num_steps, ", elapsed time", timedelta(seconds=elapsed_time))

                    save_df['question'] = strings
                    save_df['text'] = texts
                    save_df.to_csv(self.default_path + 'data/gpt_questions_df.csv')

            elapsed_time = time() - start_time
            if verbose > 0:
                print("last step from", num_steps, ", total elapsed time", timedelta(seconds=elapsed_time))

            save_df['question'] = strings
            save_df['text'] = texts
            save_df.to_csv(self.default_path + 'data/gpt_questions_df.csv')

            return texts

    ###############################
    # process texts
    ###############################

    def is_too_short(self, x):
        return len(x) < 4

    def fill_df_with_texts(self, df):

        start_time = time()
        # we do it in a iteration way
        ids = df['id'].values
        i = 0
        for id in ids:
            with open(self.default_path + 'data/texts/' + str(id) + '.txt') as reader:
                #             print("read " + str(id) + '.txt ...')
                text = reader.read()
                df.loc[df['id'] == id, ['text']] = text

                i += 1
                if i % 500 == 0:
                    elapsed_time = time() - start_time
                    print("{}/{} elapsed time: {}".format(i, len(ids), timedelta(seconds=elapsed_time)))

        elapsed_time = time() - start_time
        print("elapsed time:", timedelta(seconds=elapsed_time))

        return df

    def extract_sentences(self, df, ids):
        # Note: think about mapping functions to make it in parallel way
        all_sentences = []
        for text_id in ids:
            text = df[df['id'] == text_id]['text'].values[0]
            #         sentences = np.array(text.split('\n'))
            sentences = np.array(re.split('\; |\. |\? |\! |\n', text))
            #         print("total sentences num:", len(sentences))
            #         print("short sentences num", len(sentences[[is_too_short(x) for x in sentences]]))

            sentences = sentences[[not self.is_too_short(x) for x in sentences]]
            df.at[df.loc[df['id'] == text_id].index[0], 'split_sentences'] = json.dumps(sentences.tolist())

            #         print("short sentences after clean", len(sentences[[is_too_short(x) for x in sentences]]))
            #         print("cleaned sentences num:", len(sentences))

            #         print()
            all_sentences.extend(sentences)

        return all_sentences

    ###############################
    # closest strings to given string
    ###############################

    def find_closest_to(self, strings_embeddings, target_embedding, num_closest, verbose=0):

        start_time = time()

        # embeddings_for_strings = self.get_embedings(strings, verbose=0)
        # embeddings_for_target = self.get_embedings([target], verbose=0)

        closest = cKDTree(strings_embeddings).query(target_embedding, k=num_closest)[1]

        elapsed_time = time() - start_time
        if verbose > 0:
            print("total elapsed time:", timedelta(seconds=elapsed_time))

        return closest

    def save_articles(self, articles_by_question):
        articles_by_question.to_csv(self.default_path + 'data/articles_by_question.csv')
