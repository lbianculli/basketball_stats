import tensorflow as tf
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import itertools
import seaborn as sns; sns.set()

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')


class keras_model():
    def __init__(self, all_data, pca_components=30):
        self.all_data = all_data
        self.pca_components = pca_components

        
    def _format_data(self, label_name='home_win'):

        train_data = self.all_data.sample(frac=.8)
        test_data = self.all_data.drop(train_data.index)

        train_labels = train_data[['home_win', 'margin', 'total_points']]
        train_data = train_data.drop(train_labels, axis=1)  
        
        test_labels = test_data[['home_win', 'margin', 'total_points']]
        test_data = test_data.drop(test_labels, axis=1)
        
        train_labels = np.asarray(train_labels[label_name], dtype=np.float64)
        test_labels = np.asarray(test_labels[label_name], dtype=np.float64)

        self.train_labels = np.expand_dims(train_labels, axis=1)
        self.test_labels = np.expand_dims(test_labels, axis=1)
        self.normed_train_data = np.asarray((train_data-np.mean(train_data, axis=0)) / np.std(train_data, axis=0), dtype=np.float64)
        self.normed_test_data = np.asarray((test_data-np.mean(train_data, axis=0)) / np.std(train_data, axis=0), dtype=np.float64)
        
        if self.pca_components:
            pca = PCA(n_components=self.pca_components, whiten=True)
            self.normed_train_data = pca.fit_transform(self.normed_train_data)
            self.normed_test_data = pca.transform(self.normed_test_data)
        
        self.N_FEATURES = len(self.normed_train_data[1])
        self.N_SAMPLES = len(self.normed_train_data)
        
        
    def create_clf(self, hiddens, units, learning_rate=1e-3, label_name='home_win'): 
        self._format_data(label_name)
   
        model = keras.Sequential()
        model.add(keras.layers.Dense(units, activation=keras.layers.LeakyReLU(alpha=.25), 
                                     input_shape=[self.N_FEATURES, ]))  # input layer
        for i in range(hiddens):
            model.add(keras.layers.Dense(units, activation=keras.layers.LeakyReLU(alpha=.25)))
        model.add(keras.layers.Dense(2, activation='softmax'))  # ouput (for regression)
     
        opt = keras.optimizers.Adam(lr=learning_rate)
        model.compile(optimizer=opt, loss=['sparse_categorical_crossentropy'], metrics=['accuracy'])
        self.model = model
        
        
    def create_regression(self, hiddens, units, learning_rate=1e-3, label_name='total_points'):
        self._format_data(label_name)
   
        model = keras.Sequential()
        model.add(keras.layers.Dense(units, activation=tf.nn.leaky_relu, input_shape=[self.N_FEATURES, ]))  # input layer
        for i in range(hidden):
            model.add(keras.layers.Dense(units, activation=tf.nn.leaky_relu))
        model.add(keras.layers.Dense(1))  # ouput (for regression)
     
        opt = keras.optimizers.Adam(lr=learning_rate)
        model.compile(optimizer=opt, loss=['mean_squared_error'], metrics=['mae'])

        self.model = model
        
        
    def train_model(self, n_epochs=100, batch_size=16, plot=False):
        ''' trains keras model, returns the history object as a DF '''  # experiment with restore_best_weights/save_best_only
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)

        checkpointer = ModelCheckpoint(filepath='/home/lbianculli/nba_networks/keras/best_model_4-26', 
                                       monitor='val_loss', save_best_only=True)
    
        history = self.model.fit(
            self.normed_train_data, self.train_labels, epochs=n_epochs,validation_split=0.2,
            batch_size=batch_size, verbose=0, callbacks=[early_stop, PrintDot()])
        
        hist = pd.DataFrame(history.history)
        train_losses = hist['loss']
        val_losses = hist['val_loss']

        try:
            train_metrics = hist['acc']
            val_metrics = hist['val_acc']
        except Exception as e:
            train_metrics = hist['mae']
            val_metrics = hist['val_mae']
        
        if plot:  
            fig = plt.figure(figsize=(12,7))  
            fig.subplots_adjust(wspace=0.4)
            self.plot_loss_curve(
                n_epochs=list(hist.index), train_losses=train_losses, val_losses=val_losses)  
            self.plot_metrics_curve(
                n_epochs=list(hist.index), train_metrics=train_metrics, val_metrics=val_metrics)
        
        return hist
    
    
    def evaluate_model(self):
        self.model.evaluate(self.normed_test_data, self.test_labels)
    
    
    def architecture_search(self, hiddens, units, learning_rates, batch_sizes, n_jobs=1, label_name='home_win'): 
        '''
        iterates through architectures given collections of hyperparams:
        n_hidden: list of number of hidden layers
        n_units: list of number of units in each hidden layer
        learning_rate: list of optimizer learning rate
        n_jobs: number of CPUs to use. -1 utilizes all available CPUs
        '''
        self._format_data(label_name)
        if n_jobs == -1:
            n_jobs = cpu_count()
        results = []
        mesh = list(itertools.product(hiddens, units, learning_rates, batch_sizes))

        with Pool(n_jobs) as pool:
            results.append(pool.starmap(
                self._architecture_search, [params for params in mesh]
            ))

        return sorted(results[0])


    def _architecture_search(self, hidden, units, learning_rate, batch_size, verbose=1):  
        ''' 
        performs search of inidividual set of param values 
        validation loss as well as tuple containing param values. 
        Prints this as well if verbose is 1.
        '''
        # anyway to make this cleaner with *args?  
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

        model = keras.Sequential()
        model.add(keras.layers.Dense(units, activation=tf.nn.leaky_relu, input_shape=[self.N_FEATURES,]))  
        for i in range(hidden):
            model.add(keras.layers.Dense(units, activation=tf.nn.leaky_relu))
        model.add(keras.layers.Dense(2, activation=tf.nn.softmax))  

        opt = keras.optimizers.Adam(lr=learning_rate)
        model.compile(optimizer=opt, loss=['sparse_categorical_crossentropy'], metrics=['accuracy'])
        history = model.fit(self.normed_train_data, self.train_labels, epochs=100,
                validation_split=0.2, batch_size=batch_size, verbose=0, callbacks=[early_stop, PrintDot()])

        hist = pd.DataFrame(history.history)
        val_loss = np.mean(hist['val_loss'].tail(1))
        if verbose == 1:
            print(f'\nUnits: {units}, hidden layers: {hidden}, learning rate: {learning_rate}, \
                  batch size: {batch_size} -- Val loss: {val_loss}')

        return (val_loss, (hidden, units, learning_rate, batch_size))
    
    
    def plot_loss_curve(self, n_epochs, train_losses, val_losses=None):
        ''' Not sure the best way to incorporate these) '''
        x_ax_range = n_epochs
        plt.subplot(2, 3, 1)
        plt.title('Training and Validation Loss')
        plt.ylabel('Loss') 
        plt.xlabel('Epochs')
        plt.plot(x_ax_range, train_losses, 'r', x_ax_range, val_losses, 'g')
        
        plt.legend(['train', 'validation']);
        
        
    def plot_metrics_curve(self, n_epochs, train_metrics, val_metrics=None):
        x_ax_range = n_epochs
        plt.subplot(2, 3, 2)
        plt.title('Training and Validation Accuracy')
        plt.ylabel('Accuracy') 
        plt.xlabel('Epochs')
        plt.plot(x_ax_range, train_metrics, 'r', x_ax_range, val_metrics, 'g')
        plt.legend(['train', 'validation']);    
        
