'''
Author: Ziyu Chen
'''
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.layers import Embedding
from keras.optimizers import Adam
from keras.layers import Flatten, Conv1D, MaxPooling1D
from keras.layers import Dropout, Input, concatenate
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from utils import *


class TextCNN(object):
    def __init__(self, data, label, embedding_matrix, model = None, maxlen = 100, filter_sizes = [2,3,4,5,6,7,8],
                dropout = 0.5, hidden_units = 128, n_class = 2):
        self.data = data
        self.label = label
        self.embedding_matrix = embedding_matrix
        self.model = model
        self.maxlen = maxlen 
        self.filter_sizes = filter_sizes
        self.dropout = dropout 
        self.hidden_units = hidden_units
        self.n_class = n_class 
        self.hist = None

        
        self.X_train, self.X_eval, self.y_train, self.y_eval = train_test_split(self.data, self.label, test_size=0.3, 
                                                                                shuffle=True, random_state=8)
        self.X_eval, self.X_test, self.y_eval, self.y_test = train_test_split(self.X_eval, self.y_eval, test_size=0.33, 
                                                                              shuffle=True, random_state=4)
        
        
    def text_cnn(self):
        '''
        embedding_matrix: A collection of word vectors that are presented in the corpus. Returned from utils.py
        maxlen: The length of the padded sentences.
        '''
        if self.embedding_matrix is None:
            self.embedding_matrix = load_weights(pretrained_model = './model/wiki.en.vec', 
                                                 self_trained_model = './model/fil9.bin')

        vocab_size = len(self.embedding_matrix)
        vector_size = self.embedding_matrix.shape[1]                                    
        seq = Input(shape=[self.maxlen],name='x_seq')

        #Embedding layers
        emb = Embedding(vocab_size, vector_size, weights=[self.embedding_matrix], input_length=self.maxlen, 
                        trainable=False)(seq)
        #Conv layers
        convs = []
        filter_sizes = self.filter_sizes
        for fsz in filter_sizes:
            conv1 = Conv1D(self.hidden_units, kernel_size=fsz,activation='tanh')(emb)
            pool1 = MaxPooling1D(self.maxlen-fsz+1)(conv1)
            pool1 = Flatten()(pool1)
            convs.append(pool1)
        merge = concatenate(convs,axis=1)
        out = Dropout(self.dropout)(merge)
        output = Dense(self.n_class,activation='sigmoid')(out)
        model = Model([seq],output)
        return model
    
    
    def train(self, learning_rate = 0.0005, loss = 'binary_crossentropy', epochs = 80, batch_size = 64, plot = True): 
        # Initialize model and fit the data
        self.model = self.text_cnn()
        self.model.compile(optimizer=Adam(lr = learning_rate), loss=loss, metrics=['acc'])
        save_best = ModelCheckpoint('./saved_model.hdf', save_best_only=True, monitor='val_loss', mode='min')
        self.hist = self.model.fit(self.X_train, self.y_train, validation_data=(self.X_eval, self.y_eval),
                         verbose=2, epochs=epochs, callbacks=[save_best], batch_size=batch_size)
        
        # Plot out the training process
        if plot:
            fig = plt.figure()
            plt.subplot(2,1,1)
            plt.plot(self.hist.history['acc'])
            plt.plot(self.hist.history['val_acc'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'valid'], loc='upper left')
            plt.subplot(2,1,2)
            plt.plot(self.hist.history['loss'])
            plt.plot(self.hist.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'valid'], loc='upper left')
            plt.tight_layout()
            
        return self.model
    
    
    def test(self, model_path = './model/saved_model.hdf'):      
        if self.model is None or model_path != './model/saved_model.hdf':
            self.model = self.text_cnn()
            self.model.load_weights(filepath=model_path)
                
        # Prediction on test
        predictions = self.model.predict(self.X_test)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(self.y_test, axis=1)
        
        # Calculate all kinds of metrics
        [[TN, FP], [FN, TP]] = confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(float)
        accuracy = (TP+TN)/(TP+TN+FP+FN)
        auc = roc_auc_score(y_true, y_pred)
        specificity = TN/(FP+TN)
        precision = TP/(TP+FP)
        sensitivity = recall = TP/(TP+FN)
        fscore = 2*TP/(2*TP+FP+FN)

        print('Accuracy: ', accuracy)
        print('F1 score: ', fscore)
        print('AUC: ', auc)
        print('Precision: ', precision)
        print('Recall: ', recall)

        
        
        
