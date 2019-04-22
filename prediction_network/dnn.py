import tensorflow as tf
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import seaborn as sns; sns.set()

class deep_net():
    ''' 
    creates deep model with Tensorflow
    Think I would rather do training/testing instead of all data. can then split off validation set in class
    
    '''
    def __init__(self, train_data, all_data, 
                 learning_rate=1e-3, pca_components=None, reuse=False, logdir=None):
        self.all_data = all_data
        self.learning_rate = learning_rate
        self.pca = pca
        self.reuse = reuse
        self.LOGDIR = logdir
        
        
    def _format_data(self, label_name='home_win')
        if label_name = 'home_win':
            self.regression = False
            
        pre_split_data = self.all_data.sample(frac=.8)
        test_data = self.all_data.drop(pre_split_data.index)
        train_data = pre_split_data.sample(frac=.8)
        valid_data = pre_split_data.drop(train_data.index)

        train_labels = train_data[['home_win', 'margin', 'total_points']]
        train_data = train_data.drop(train_labels, axis=1)  # these are the three possible labels
        test_labels = test_data[['home_win', 'margin', 'total_points']]
        test_data = test_data.drop(test_labels, axis=1)
        valid_labels = valid_data[['home_win', 'margin', 'total_points']]
        valid_data = valid_data.drop(valid_labels, axis=1)
        train_labels = np.asarray(train_labels[label_name], dtype=np.float64)
        test_labels = np.asarray(test_labels[label_name], dtype=np.float64)
        valid_labels = np.asarray(valid_labels[label_name], dtype=np.float64)
        
        self.train_labels = np.expand_dims(train_labels, axis=1)
        self.test_labels = np.expand_dims(test_labels, axis=1)
        self.valid_labels = np.expand_dims(valid_labels, axis=1)
        self.normed_train_data = np.asarray((train_data-np.mean(train_data, axis=0)) / np.std(train_data, axis=0), dtype=np.float64)
        self.normed_test_data = np.asarray((test_data-np.mean(train_data, axis=0)) / np.std(train_data, axis=0), dtype=np.float64)
        self.normed_valid_data = np.asarray((valid_data-np.mean(train_data, axis=0)) / np.std(train_data, axis=0), dtype=np.float64)
        
        if self.pca:
            pca = PCA(n_components=self.pca_components, whiten=True)
            self.normed_train_data = pca.fit_transform(normed_train_data)
            self.normed_test_data = pca.transform(normed_test_data)
            self.normed_valid_data = pca.transform(normed_valid_data)

        self.N_FEATURES = len(normed_train_data[1])
        self.N_SAMPLES = len(normed_train_data)
        
    def _activation_summary(self, _x):
        ''' Creates summaries for activations. Returns nothing. '''
        tensor_name = _x.op.name
        tf.summary.histogram(tensor_name + '/activations', _x)
        tf.summary.scalar(tensor_name + '/sparsity', 
                          tf.nn.zero_fraction(_x))
        
        
    def _create_network(self, n_units=4):
        '''
        setup graph for network. returns output depending on if it is a classification of regression
        n_units: number of units in each hidden layer
        '''
        self.x = tf.cast(tf.placeholder(dtype=tf.float32, shape=[None, self.N_FEATURES], name='input'), tf.float32)
        self.y = tf.cast(tf.placeholder(dtype=tf.float32, shape=[None, 1], name='labels'), tf.float32)

        with tf.variable_scope('deep_network') as scope:
            out1 = tf.layers.dense(x, units=n_units, activation=tf.nn.relu)
            self._activation_summary(out1)
            out2 = tf.layers.dense(out1, units=n_units, activation=tf.nn.relu)
            self._activation_summary(out2)
            out3 = tf.layers.dense(out2, units=n_units, activation=tf.nn.relu)
            self._activation_summary(out3)
        
        if self.regression:
            with tf.variable_scope('predictions') as scope:
                preds = tf.layers.dense(out3, units=1, activation=None)
                self._activation_summary(preds)
        else: 
            with tf.variable_scope('predictions') as scope:
                preds = tf.layers.dense(out3, units=2, activation=None)
                self._activation_summary(preds)
        
        return preds
    
    
    def _train_regression(self, n_epochs=50, batch_size=32, verbose=True)
        if self.reuse is False:
            tf.reset_default_graph()
        preds = self._create_network()

        with tf.variable_scope('loss') as scope:
            loss = tf.losses.mean_squared_error(labels=self.y, predictions=preds)
 
        with tf.variable_scope('training') as scope: 
            optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            summ = tf.summary.merge_all()
            saver = tf.train.Saver()
            summary_writer = tf.summary.FileWriter(self.LOGDIR + str(lr))
            summary_writer.add_graph(sess.graph)
            train_losses = []
            valid_losses = []
            test_losses = []

            for epoch in range(1, n_epochs+1):
                for batch in range(self.N_SAMPLES // batch_size): 
                    batch_x = self.normed_train_data[batch*batch_size: min((batch+1)*batch_size, self.N_SAMPLES)]
                    batch_y = self.train_labels[batch*batch_size: min((batch+1)*batch_size, self.N_SAMPLES)]

                    train_opt = sess.run(optimizer, feed_dict={self.x: batch_x, y: batch_y})
                    train_loss, s = sess.run([loss, summ], feed_dict={self.x: batch_x, y: batch_y})  
                    summary_writer.add_summary(s, batch)

                valid_loss = sess.run(loss, feed_dict={self.x: self.valid_data, self.y: self.valid_labels}) 
                train_losses.append(train_loss)
                valid_losses.append(valid_loss)
                if epoch % 10 == 0 and verbose:
                    print(f'Epoch {epoch} train loss: {train_loss} \nValidation loss: {valid_loss}')
        print(f'\nTraining Complete! Run `tensorboard --logdir={self.LOGDIR+str(lr)}` to see the results.')
                                      
                                      
    def _train_classification(self, n_epochs=50, batch_size=32, verbose=True):
        if self.reuse is False:
            tf.reset_default_graph()
        preds = self._create_network()

        with tf.variable_scope('loss') as scope:
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=preds))

        with tf.variable_scope('training') as scope: 
            optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
                                  
        with tf.variable_scope('accuracy') as scope:
            correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(self.y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            summ = tf.summary.merge_all()
            saver = tf.train.Saver()
            summary_writer = tf.summary.FileWriter(self.LOGDIR + str(lr))
            summary_writer.add_graph(sess.graph)
            self.train_losses = []
            self.train_accs = []
            self.valid_losses = []
            self.valid_accs = []

            for epoch in range(1, n_epochs+1):
                for batch in range(self.N_SAMPLES // batch_size): 
                    batch_x = self.normed_train_data[batch*batch_size: min((batch+1)*batch_size, self.N_SAMPLES)]
                    batch_y = self.train_labels[batch*batch_size: min((batch+1)*batch_size, self.N_SAMPLES)]
                    
                    train_opt = sess.run(optimizer, feed_dict={self.x: batch_x, self.y: batch_y})
                    train_loss, train_acc, s = sess.run([loss, acc, summ], feed_dict={self.x: batch_x, self.y: batch_y})  
                    summary_writer.add_summary(s, batch)

                valid_loss = sess.run(loss, feed_dict={self.x: self.valid_data, self.y: self.valid_labels}) 
                self.train_losses.append(train_loss)
                self.valid_losses.append(valid_loss)
                self.train_accs.append(train_acc)
                self.valid_accs.append(valid_acc)
                if epoch % 10 == 0 and verbose:
                    print(f'Epoch {epoch} train loss: {train_loss} \nValidation loss: {valid_loss}')
        print(f'\nTraining Complete! Run `tensorboard --logdir={self.LOGDIR+str(lr)}` to see the results.')
        
        
    def plot_loss_curve(self, n_epochs=50):
        ''' Not sure the best way to incorporate these) '''
        x = range(n_epochs)
        fig = plt.figure(figsize=(9,6))
        plt.title('Training and Validation Loss')
        plt.ylabel('Loss') 
        plt.xlabel('Epochs')
        plt.plot(x, self.train_losses, 'r', x, self.valid_losses, 'g')
        plt.legend();
        
    def plot_accuracy_curve(self, n_epochs=50):
        x = range(n_epochs)
        fig = plt.figure(figsize=(9,6))
        plt.title('Training and Validation Accuracy')
        plt.ylabel('Accuracy') 
        plt.xlabel('Epochs')
        plt.plot(x, self.train_accs, 'r', x, self.valid_accs, 'g')
        plt.legend();
        
