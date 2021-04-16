# car_reviews_utils.py
# Utility code for the car reviews assignment

#### Import libraries
import re
from nltk.stem.snowball import EnglishStemmer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from nltk.stem import WordNetLemmatizer 
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec

#### Part 1 functions
def clean_text_data(df, var, stopwords):
    """
    Function to clean a text column in a Pandas DataFrame
    - converts string to lowercase
    - removes stopwords
    - removes numbers and special characters
    - removes multiple spaces
    """
    df_copy = df.copy()
    df_copy[var] = df_copy[var].apply(lambda x: x.lower())
    df_copy[var] = df_copy[var].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))
    df_copy[var] = df_copy[var].apply(lambda x: re.sub('[^a-z ]', "", x))
    df_copy[var] = df_copy[var].apply(lambda x: ' '.join(x.split()))
    
    return df_copy

def stem_docs(docs):
    """Return stemmed documents"""
    stemmer = EnglishStemmer()
    return [" ".join([stemmer.stem(word) for word in sentence.split(" ")]) for sentence in docs]

def before_and_after_stemming(string, vocab, vocab_stemmed):
    """Prints vocabulary words beginning in a sequence of characters before and after stemming"""
    r = re.compile("^" + string)
    vocab_sample = list(filter(r.match, vocab))
    vocab_sample_stem = list(filter(r.match, vocab_stemmed))

    print(f'Stemming example for words starting with "{string}"')
    print('Original vocabulary')
    print(vocab_sample)

    print('Stemmed vocabulary')
    print(vocab_sample_stem, '\n')

def plot_word_freq(x_vec, word_names, n_words=10, title='Word Frequency', figsize=(10, 8), color='orange'):
    """Plots word frequencies from a trained count vectorizer"""
    vocab_df = pd.DataFrame()
    vocab_df['words'] = word_names
    vocab_df['counts'] = x_vec.sum(axis=0)
    vocab_df.sort_values('counts', inplace=True)

    plt.figure(figsize=figsize)
    plt.title(title)
    plt.barh(vocab_df['words'][-n_words:], vocab_df['counts'][-n_words:], color=color, alpha=0.6)
    plt.show()
    
def print_accuracy(y_train, y_train_pred, y_test, y_test_pred):
    """Prints the training and test set accuracy of the classifier"""
    print(f'training set accuracy {round(accuracy_score(y_train, y_train_pred), 4)}')
    print(f'test set accuracy {round(accuracy_score(y_test, y_test_pred), 4)}')
    
def labelled_confusion_matrix(y, y_pred, prop=False):
    """Returns a labelled confusion matrix"""
    matrix = pd.DataFrame(confusion_matrix(y, y_pred))
    matrix.columns = ['Predicted:0', 'Predicted:1']
    matrix['Total'] = matrix['Predicted:0'] + matrix['Predicted:1']
    matrix = matrix.append(matrix.sum(), ignore_index=True)
    matrix.index = ['Actual:0', 'Actual:1', 'Total']

    if prop is True:
        matrix = round(matrix / matrix.iloc[2, 2] , 4)

    return matrix

def print_metrics(y_train, y_train_pred, y_test, y_test_pred):
    """Prints performance precision, recall, and f1 scores"""
    print('training set performance:')
    print(f'training precision score: {round(precision_score(y_train, y_train_pred), 4)}')
    print(f'training recall score: {round(recall_score(y_train, y_train_pred), 4)}')
    print(f'training f1 score: {round(f1_score(y_train, y_train_pred), 4)}')

    print('\ntest set performance:')
    print(f'testing precision score: {round(precision_score(y_test, y_test_pred), 4)}')
    print(f'testing recall score: {round(recall_score(y_test, y_test_pred), 4)}')
    print(f'testing f1 score: {round(f1_score(y_test, y_test_pred), 4)}')


# Part 2 functions & classes
def get_wordnet_pos(treebank_tag):
    """Returns WordNet part of speech"""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None 
    
def tag_and_lemmatize(tokens, lemmatizer):
    """Tag and lemmatize a list of word tokens"""
    transformed_docs = []
    tagged = pos_tag(tokens)
    for word, tag in tagged:
        wntag = get_wordnet_pos(tag)
        if wntag is None:
            lemma = lemmatizer.lemmatize(word) 
        else:
            lemma = lemmatizer.lemmatize(word, pos=wntag) 
        transformed_docs.append(lemma)
    return transformed_docs

def lemmatize_docs(docs):
    """Return lemmatized documents"""
    lemmatizer = WordNetLemmatizer() 
    #lemmed = [[lemmatizer.lemmatize(word) for word in word_tokenize(doc)] for doc in docs]
    lemmed = [tag_and_lemmatize(word_tokenize(doc), lemmatizer) for doc in docs]
    return [" ".join(x) for x in lemmed]

def feature_vectorisation(x_train, x_test, stem=True, count_vectorizer=True, min_df=1, ngram_range=(1, 1), scale_data=False):
    """Applies varying settings for feature vectorisation"""
    if stem:
        x_train = stem_docs(x_train)
        x_test = stem_docs(x_test)
    else:
        x_train = lemmatize_docs(x_train)
        x_test = lemmatize_docs(x_test)

    if count_vectorizer is True:
        vectorizer = CountVectorizer(min_df=min_df, ngram_range=ngram_range)
    else:
        vectorizer = TfidfVectorizer(min_df=min_df, ngram_range=ngram_range)
        
    vectorizer.fit(x_train)

    x_train_vec = vectorizer.transform(x_train).toarray()
    x_test_vec = vectorizer.transform(x_test).toarray()
    
    if scale_data:
        scaler = StandardScaler()
        x_train_vec = scaler.fit_transform(x_train_vec)
        x_test_vec = scaler.transform(x_test_vec)
        
    return x_train_vec, x_test_vec

class MonteCarloCV:
    """
    Monte Carlo Cross Validation.
    """
    def __init__(self, models, pct_train=0.7, number_of_runs=100, random_seed=42):

        self.models = models
        self.pct_train = pct_train
        self.number_of_runs = number_of_runs
        self.random_seed = random_seed
        self.results_dict = {}
        self.train_meta_scores = []
        self.val_meta_scores = []
        self.train_meta_costs = []
        self.val_meta_costs = []
        
    def train(self, x, y, feature_processing=None, verbose=False, verbose_n=10):
        """
        Train the models.
        """
        n_train = int(len(x) * self.pct_train)
        
        for m, model in enumerate(self.models):
            if verbose:
                print(f'Model {m}')
            train_scoring = []
            val_scoring = []
            train_costs_model = []
            val_costs_model = []
            
            x_copy = x.copy()
            y_copy = y.copy()

            seed = self.random_seed
            
            for i in range(self.number_of_runs):
                
                np.random.seed(seed)
                np.random.shuffle(x_copy)
                np.random.seed(seed)
                np.random.shuffle(y_copy)
                seed += 1
                
                x_train = x_copy[:n_train]
                x_val = x_copy[n_train:]

                y_train = y_copy[:n_train]
                y_val = y_copy[n_train:]
                
                if feature_processing is not None:
                    x_train, x_val = feature_processing(x_train, x_val)
 
                model.fit(x_train, y_train)
    
                train_predictions = model.predict(x_train)
                val_predictions = model.predict(x_val)
                
                train_accuracy = np.count_nonzero(train_predictions == y_train)/len(y_train)
                val_accuracy = np.count_nonzero(val_predictions == y_val)/len(y_val)
                
                train_scoring.append(train_accuracy)
                val_scoring.append(val_accuracy)
                
                if verbose and i % verbose_n == 0:
                    print(f'Iteration {i} - cumulative average accuracy: {np.mean(val_scoring)}')
                    
            if verbose:
                print(f'Model {m} average accuracy: {np.mean(val_scoring)} \n')

            self.results_dict[m] = (np.mean(val_scoring), np.std(val_scoring))
            self.train_meta_scores.append(train_scoring)
            self.val_meta_scores.append(val_scoring)
            self.train_meta_costs.append(train_costs_model)
            self.val_meta_costs.append(val_costs_model)
   
    def train_params(self, x, y, feature_processing=None, params=None, verbose=False, verbose_n=10):
        """
        Train the models with user supplied parameters
        """
        n_train = int(len(x) * self.pct_train)
        c=0
        
        for m, model in enumerate(self.models):
            if verbose:
                print(f'Model {m}')
            train_scoring = []
            val_scoring = []
            train_costs_model = []
            val_costs_model = []

            x_copy = x.copy()
            y_copy = y.copy()

            seed = self.random_seed

            param = params[m]
            stem = param['stem']
            count_vectorizer = param['count_vectorizer']
            min_df = param['min_df']
            ngram_range = param['ngram_range']
            
            for i in range(self.number_of_runs):

                np.random.seed(seed)
                np.random.shuffle(x_copy)
                np.random.seed(seed)
                np.random.shuffle(y_copy)
                seed += 1

                x_train = x_copy[:n_train]
                x_val = x_copy[n_train:]

                y_train = y_copy[:n_train]
                y_val = y_copy[n_train:]

                # Feature processing
                x_train, x_val = feature_processing(x_train, x_val, stem=stem, count_vectorizer=count_vectorizer,
                                                    min_df=min_df, ngram_range=ngram_range)

                model.fit(x_train, y_train)

                train_predictions = model.predict(x_train)
                val_predictions = model.predict(x_val)

                train_accuracy = np.count_nonzero(train_predictions == y_train)/len(y_train)
                val_accuracy = np.count_nonzero(val_predictions == y_val)/len(y_val)

                train_scoring.append(train_accuracy)
                val_scoring.append(val_accuracy)

                if verbose and i % verbose_n == 0:
                    print(f'Iteration {i} - cumulative average accuracy: {np.mean(val_scoring)}')

            if verbose:
                print(f'Model {m} - average accuracy: {np.mean(val_scoring)} \n')

            self.results_dict[m] = (np.mean(val_scoring), np.std(val_scoring))
            self.train_meta_scores.append(train_scoring)
            self.val_meta_scores.append(val_scoring)
            self.train_meta_costs.append(train_costs_model)
            self.val_meta_costs.append(val_costs_model)
       
    def plot_scores(self, colors=None, labels=None, bins=20, xlim=(0.0, 1.0)):
        """
        Plot the density functions of the model scores.
        """

        fig = plt.figure(figsize=(10, 6))
        ax = plt.gca()
        plt.title(f'Monte Carlo CV Density Function - Results over {self.number_of_runs} runs',
                  fontsize=14, weight='bold')
        for m, scores in enumerate(self.val_meta_scores):
            if labels is None:
                  label = ''
            else:
                label = labels[m] + ' - mean: ' + (str(round(np.mean(scores), 4))) + ' stdev: ' +(str(round(np.std(scores), 4)))
            if colors is None:
                sns.kdeplot(scores, label=label, fill=True, alpha=0.25)
            else:
                sns.kdeplot(scores, color=colors[m], label=label, fill=True, alpha=0.25)
        plt.xlabel('validation accuracy')
        plt.ylabel('frequency')
        plt.xlim(xlim)
        if labels is not None:
            plt.legend(loc='upper left', bbox_to_anchor=(0.0, -0.1), frameon=False)
        plt.show() 
        
def label_sentences(corpus, label_type):
    """
    Create document tags
    """
    labeled = []
    for i, v in enumerate(corpus):
        label = label_type + '_' + str(i)
        labeled.append(TaggedDocument(v.split(), [label]))
    return labeled

def get_vectors(model, documents):
    """
    Get vectors from trained Doc2Vec model
    """
    corpus_size = len(documents)
    vector_size = model.vector_size
    vectors = np.zeros((corpus_size, vector_size))
    
    for i, doc in enumerate(documents):
        vectors[i] = model.infer_vector(doc[0])
    return vectors

def feature_vectorisation_d2v(x_train, x_test, dm=0, vector_size=300, window=15, min_count=5, 
                              sample=10e-5, alpha=0.025, epochs=20):
    """Create Doc2Vec vectors"""
    x_train_tagged = label_sentences(x_train, 'Train')
    x_test_tagged = label_sentences(x_test, 'Test')

    d2v_model = Doc2Vec(dm=dm, vector_size=vector_size, window=window, min_count=min_count, 
                        sample=sample, alpha=alpha, epochs=epochs)
    d2v_model.build_vocab(x_train_tagged)
    d2v_model.train(x_train_tagged, total_examples=len(x_train_tagged), epochs=d2v_model.epochs)
    
    train_vectors_dbow = get_vectors(d2v_model, x_train_tagged)
    test_vectors_dbow = get_vectors(d2v_model, x_test_tagged)
    
    return train_vectors_dbow, test_vectors_dbow

class StackedClassifier:
    """
    Builds a stacked classifier
    """
    def __init__(self, estimators, random_seed=42):
        self.estimators = estimators
        self.n_estimators = len(estimators)
        self.random_seed = random_seed
    
    def fit(self, x, y, test_pct=0.2, l1_pct=0.7, verbose=False):
        """Fit models"""
        
        n_train = int(x.shape[0] * (1- test_pct))
        l1_train = int(n_train * l1_pct)
        
        y = np.array(y)
        
        # Take copies and shuffle the data
        x_copy = x.copy()
        y_copy = y.copy()
        
        np.random.seed(self.random_seed)
        np.random.shuffle(x_copy)
        np.random.seed(self.random_seed)
        np.random.shuffle(y_copy)
        
        # Create datasets
        x_train_l1 = x_copy[:l1_train]
        y_train_l1 = y_copy[:l1_train]
        x_train_l2 = x_copy[l1_train:n_train]
        y_train_l2 = y_copy[l1_train:n_train]
        x_test = x_copy[n_train:]
        y_test = y_copy[n_train:]

        for i, estimator in enumerate(self.estimators):
            if i < self.n_estimators - 1:
                estimator.fit(x_train_l1, y_train_l1)
                train_l1_pred = estimator.predict(x_train_l1)
                train_l2_pred = estimator.predict(x_train_l2)

                if verbose:
                    print(f'Model {i} performance:')
                    print_metrics(y_train_l1, train_l1_pred, y_train_l2, train_l2_pred)
                    
            else:
                l2_train_preds = self.create_l2_predictions(x_train_l2) 
                l2_test_preds = self.create_l2_predictions(x_test) 

                self.estimators[i].fit(l2_train_preds, y_train_l2)
                train_pred = self.estimators[i].predict(l2_train_preds)
                test_pred = self.estimators[i].predict(l2_test_preds)

                if verbose:
                    print(f'Meta model performance:')
                    print_metrics(y_train_l2, train_pred, y_test, test_pred)
                             
    def create_l2_predictions(self, x_l2):
            l2_preds = np.zeros((x_l2.shape[0], self.n_estimators-1))
                
            for i, estimator in enumerate(self.estimators[:-1]):
                l2_preds[:, i] = estimator.predict(x_l2)
                
            return l2_preds
                    
    def predict(self, x):
        """Predict on new data"""
        preds = np.zeros(x.shape[0])
        l2_preds = self.create_l2_predictions(x) 
        preds = self.estimators[-1].predict(l2_preds)
        preds = np.rint(preds)
        
        return preds
    