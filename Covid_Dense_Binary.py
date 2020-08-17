
# coding: utf-8

# In[ ]:

import pandas as pd
from keras.models import Model
from keras.layers import Input, Embedding, Dense, GRU, Bidirectional, TimeDistributed, Lambda, Dropout, LSTM, Flatten
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.utils.np_utils import to_categorical
import keras.backend as K
from keras import regularizers
from keras.layers import Lambda
from keras.layers import Flatten
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.core import Activation, Dense
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
import numpy as np
import pickle
import os
from glob import glob


from .attention import Attention

DENSE_SIZE = 128
Special_value= -10


class HAN(Model):
    def __init__(self, max_sent_length=5,
                 max_sent_num=5, max_doc_num=5, word_embed_dim=100, sent_embed_dim=100, doc_embed_dim=5):
        """Implementation of Hierarchical Attention Networks for document classification.

        Args:
            embedding_matrix: embedding matrix used to represent words.
            max_sent_length: int, maximum number of words per sentence, default is 100.
            max_sent_num: int, maximum number of sentences accepted, default is 10.
            word_embed_dim: int, dimension of word encoder, default is 100.
            sent_embed_dim: int, dimension of sentence encoder, default is 100.
        
            ************************************
            max_doc_num: int, maximum number of documents per patient, default is 5.
            doc_embed_dim: int, dimension of document encoder, default is 300.  
            
        """
        #self.embedding_matrix = embedding_matrix
        self.max_sent_length = max_sent_length
        self.max_sent_num = max_sent_num
        self.word_embed_dim = word_embed_dim
        self.sent_embed_dim = sent_embed_dim
        self.max_doc_num=max_doc_num
        self.doc_embed_dim=doc_embed_dim

        super(HAN, self).__init__(name='han')
        self.build_model()

    
 

    def build_model(self):
        """Build the embed and encode models for word and sentence.

        For the word model, the sequence of Layers is: Embedding ->
        Bidirectional(GRU) -> TimeDistributed(Dense) -> Attention

        For the sentence model, it takes the word level model as input
        for TimeDistributed Layer to make sentence encoder. And the
        sequence is: TimeDistributed(WordModel) -> Bidirectional(GRU)
        -> TimeDistributed(Dense) -> Attention -> Dense

        There is no output, but will save the word and sentence models.
        
        text_input = Input(shape=(self.max_sent_num, self.max_sent_length))
        # encode sentences into a single vector per sentence
        self.model_word = self.build_word_encoder()
        # time distribute word model to accept text input
        sent_encoder = TimeDistributed(self.model_word)(text_input)

        doc_att = self.build_sent_encoder(sent_encoder)
        
        
        """
        doc_input = Input(shape=(self.max_sent_num ,self.max_sent_length,512), dtype='float32')
        doc_in=Flatten()(doc_input)
        
        #masked3=Masking(mask_value=Special_value)(doc_input)
        
 #       self.model_sent = self.build_sent_encoder()
        
 #      doc_encoder= TimeDistributed(self.model_sent)(doc_in)
        
  #      document_att= self.build_doc_encoder(doc_encoder)
        dense= Dense(DENSE_SIZE,activation='softmax')(doc_in)
        #doc_att = self.build_sent_encoder(sent_encoder)
        # dense the output to 2 because the result is a binary classification.
        output_tensor = Dense(2, activation='softmax', name='classification')(dense)
        # Create Sentence-level Model
        self.model = Model(doc_input, output_tensor)

    def print_summary(self):
        """Print the model summary for both word and sentence level model.
        """
        print("Word Level")
        self.model_word.summary()
        
        print("Sent Level")
        self.model_sent.summary()

        print("Doc Level")
        self.model.summary()

    def train_model(self, checkpoint_path, X_train, y_train, X_val, y_val, X_test, y_test,
                    optimizer='adagrad', loss='categorical_crossentropy',
                    metric='acc', monitor='val_loss', batch_size=20, epochs=10):
        """Train the HAN model.

        Args:
            checkpoint_path: str, the path to save checkpoint file.
            X_train: training dataset.
            y_train: target of training dataset.
            X_test: testing dataset.
            y_test: target of testing dataset.
            optimizer: optimizer for compiling, default is adagrad.
            loss: loss function, default is categorical_crossentropy.
            metric: measurement metric, default is acc (accuracy).
            monitor: monitor of metric to pick up best weights, default is val_loss.
            batch_size: batch size, default is 20.
            epochs: number of epoch, default is 10.
        """
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[metric]
        )
        checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=monitor,
            verbose=1, save_weights_only=True,save_best_only=True
        )
        self.model.fit(
            X_train, y_train, batch_size=batch_size, epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=[checkpoint],shuffle=True
         )

        results=self.model.evaluate(X_test, y_test, batch_size=10)
        yhat = self.model.predict(X_test)
        y_test_non_category = [ np.argmax(t) for t in y_test ]
        y_predict_non_category = [ np.argmax(t) for t in yhat ]


        conf_mat = confusion_matrix(y_test_non_category, y_predict_non_category)
        # confusion matrix
   #     matrix = confusion_matrix(y_test[:,1], yhat_classes)
        print('test loss, test acc:', results)
        return results[1],conf_mat


    #   def show_word_attention(self, x):
        """Show the prediction of the word level attention.

        Args:
            x: the input array with size of (max_sent_length,).

        Returns:
            Attention weights.
        """
 #       att_layer = self.model_word.get_layer('word_attention')
 #       prev_tensor = att_layer.input

        # Create a temporary dummy layer to hold the
        # attention weights tensor
 #       dummy_layer = Lambda(
 #           lambda x: att_layer._get_attention_weights(x)
 #       )(prev_tensor)

 #       return Model(self.model_word.input, dummy_layer).predict(x)

    def show_sent_attention(self, x):
        """Show the prediction of the sentence level attention.

        Args:
            x: the input array with the size of (max_sent_num, max_sent_length).

        Returns:
            Attention weights.
        """
        att_layer = self.model_sent.get_layer('sent_attention')
        prev_tensor = att_layer.input

        dummy_layer = Lambda(
            lambda x: att_layer._get_attention_weights(x)
        )(prev_tensor)

        return Model(self.model_sent.input, dummy_layer).predict(x)
    
    def show_doc_attention(self, x):
        """Show the prediction of the sentence level attention.

        Args:
            x: the input array with the size of (max_sent_num, max_sent_length).

        Returns:
            Attention weights.
        """
        att_layer = self.model.get_layer('doc_attention')
        prev_tensor = att_layer.input

        dummy_layer = Lambda(
            lambda x: att_layer._get_attention_weights(x)
        )(prev_tensor)

        return Model(self.model.input, dummy_layer).predict(x)

    @staticmethod
 #   def word_att_to_df(sent_tokenized_review, word_att):
        
        # remove the trailing dot
 #       ori_sents = [i.rstrip('.') for i in sent_tokenized_review]
        # split sentences into words
 #       ori_words = [x.split() for x in ori_sents]
        # truncate attentions to have equal size of number of words per sentence
 #       truncated_att = [i[-1 * len(k):] for i, k in zip(word_att, ori_words)]

        # create word attetion pair as dictionary
 #       word_att_pair = []
 #       for i, j in zip(truncated_att, ori_words):
 #           word_att_pair.append(dict(zip(j, i)))

 #       return pd.DataFrame([(x, y) for x, y in zip(word_att_pair, ori_words)],
 #                           columns=['word_att', 'transcript'])

 #   @staticmethod
    def sent_att_to_df(sent_tokenized_reviews, sent_att):
        """Convert the sentence attention arrays into pandas dataframe.

        Args:
            sent_tokenized_reviews: sent tokenized reviews, if original input is a Series,
                that means at least Series.apply(lambda x: sent_tokenize(x)) has to be
                executed beforehand.
            sent_att: sentence attention weight obtained from self.show_sent_attetion.

        Returns:
            df: pandas.DataFrame, contains original reviews column and sent_att column,
                and sent_att column is a list of dictionaries in which sentence as key
                while corresponding weight as value.
        """
        # create reviews attention pair list
        reviews_atts = []
        for review, atts in zip(sent_tokenized_reviews, sent_att):
            review_list = []
            for sent, att in zip(review, atts):
                # each is a list of dictionaries
                review_list.append({sent: att})
            reviews_atts.append(review_list)
        return pd.DataFrame([(x, y) for x, y in zip(reviews_atts, sent_tokenized_reviews)],
                            columns=["sent_att", "transcript"])
    
    
    
    def doc_att_to_df(reviews, doc_att):
        docs_atts = []
        for review, atts in zip(reviews, doc_att):
            doc_list = []
            for doc, att in zip(review, atts):
                doc_list.append({doc: att})
            docs_atts.append(doc_list)
                
        
        return pd.DataFrame([(x, y) for x, y in zip(docs_atts,reviews)],
                            columns=["doc_att", "transcript"])
    
    

