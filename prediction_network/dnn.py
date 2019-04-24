import tensorflow as tf
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import itertools
import seaborn as sns; sns.set()


class keras_clf():
    def __init__(self, all_data, learning_rate=1e-3, pca_components=30):
        self.all_data = all_data
        self.pca_components = pca_components
        self.learning_rate = learning_rate
        
        
    def _format_data(self, label_name='home_win'):
        train_data = self.all_data.sample(frac=.8)
        test_data = self.all_data.drop(train_data.index)

        train_labels = train_data[['home_win', 'margin', 'total_points']]
        train_data = train_data.drop(train_labels, axis=1)  
        
        test_labels = test_data[['home_win', 'margin', 'total_points']]
        test_data = test_data.drop(test_labels, axis=1)
        
        train_labels = np.asarray(train_labels[label_name], dtype=np.float64)
        test_labels = np.asarray(test_labels[label_name], dtype=np.float64)

        train_labels = np.expand_dims(train_labels, axis=1)
        self.test_labels = np.expand_dims(test_labels, axis=1)
        normed_train_data = np.asarray((train_data-np.mean(train_data, axis=0)) / np.std(train_data, axis=0), dtype=np.float64)
        self.normed_test_data = np.asarray((test_data-np.mean(train_data, axis=0)) / np.std(train_data, axis=0), dtype=np.float64)
        
        if self.pca_components:
            pca = PCA(n_components=self.pca_components, whiten=True)
            normed_train_data = pca.fit_transform(normed_train_data)
            self.normed_test_data = pca.transform(self.normed_test_data)
            
        sm = SMOTE()
        self.normed_train_data = pca.fit_transform(normed_train_data)
        
        self.N_FEATURES = len(normed_train_data[1])
        self.N_SAMPLES = len(normed_train_data)
        
        
    def create_model(self, n_hidden, n_units, label_name='home_win'):
        self._format_data(label_name)
    
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(2, activation=tf.nn.leaky_relu, input_shape=(self.N_FEATURES,)))
        for l in range(n_hidden):
            model.add(tf.keras.layers.Dense(2, activation=tf.nn.leaky_relu))
        model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))
        model.compile(optimizer='Adam', loss=['sparse_categorical_crossentropy'], metrics=['accuracy'])
        
        self.model = model
        
        
    def train_model(self, epochs=100, batch_size=32, plot=False):
        ''' Still need: graphs -- how to plot the train/val losses/accs? Also I probably need subplots '''
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

        history = self.model.fit(self.normed_train_data, self.train_labels, epochs=epochs,
                    validation_split=0.2, verbose=0, callbacks=[early_stop, PrintDot()])
        
        train_losses = history['loss']  # make sure these col names are correct
        val_losses = history['val_loss']
        train_accs = history['loss']
        val_accs = history['val_acc']
        
        if plot:  
#             fig = plt.figure(figsize=(9,6))  # create fig in here?
            self.plot_loss_curve(n_epochs, train_losses, val_losses)
            self.plot_accuracy_curve(n_epochs, train_accs, val_accs)
        
        return history
    
    
    def architecture_search(self, hidden, units, learning_rates): 
        '''
        iterates through architectures given collections of hyperparams:
        n_hidden: list of number of hidden layers
        n_units: list of number of units in each hidden layer
        learning_rate: list of optimizer learning rate
        '''
        results = []
        mesh = list(itertools.product(n_units, n_hidden, learning_rates))
        for param_combo in mesh:
            for i in param_combo:
                units = i[0]
                hidden_layers = i[1]
                learning_rate = [2]

                results.append(self._architecture_search(hidden_layers, units, learning_rate), param_combo) 
                
        return results
    
    
    def _architecture_search(self, hidden, units, learning_rate):  # create function that performs search, wrap in another that iterates via vectorize and ix_
        ''' performs search of inidividual set of param values '''
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

        model = keras.layers.Sequential()
        model.add(keras.layers.Dense(units, activation=tf.nn.leaky_relu, input_shape=[self.N_FEATURES, ]))  # input layer
        for i in range(hidden):
            model.add(keras.layers.Dense(units, activation=tf.nn.leaky_relu))
        model.add(keras.layers.Dense(1))  # ouput (for regression)
     
        opt = optimizers.Adam(lr=learning_rate, decay=None)
        model.compile(optimizer=opt, loss=['mean_squared_error'], metrics=['mae'])
        history = model.fit(self.normed_train_data, self.train_labels, epochs=200,
                validation_split=0.2, verbose=0, callbacks=[early_stop, PrintDot()])

        val_loss = np.mean(history.history['val_loss'][-1])
        print(f'Units: {units}, hidden_layers: {hidden}, learning_rate: {learning_rate} -- Final Val loss: {val_loss}')
        
        return val_loss
    
    
    def plot_loss_curve(self, n_epochs, train_losses, val_losses=None):
        ''' Not sure the best way to incorporate these) '''
        x_ax_range = range(n_epochs)
#         fig = plt.figure(figsize=(9,6))
        plt.subplot(2, 3, 1)
        plt.title('Training and Validation Loss')
        plt.ylabel('Loss') 
        plt.xlabel('Epochs')
        plt.plot(x_ax_range, train_losses, 'r', x_ax_range, val_losses, 'g')
        plt.legend();
        
        
    def plot_accuracy_curve(self, n_epochs, train_accs, val_accs=None):
        x_ax_range = range(n_epochs)
#         fig = plt.figure(figsize=(9,6))
         plt.subplot(2, 3, 2)
        plt.title('Training and Validation Accuracy')
        plt.ylabel('Accuracy') 
        plt.xlabel('Epochs')
        plt.plot(x_ax_range, train_accs, 'r', x_ax_range, val_accs, 'g')
        plt.legend();        
