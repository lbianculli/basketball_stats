class deep_net(train_data, train_labels, valid_data, valid_labels, test_data, test_labels, scope,
                lr=1e-3, n_epochs=50, batch_size=32, reuse=False, verbose=True):
    ''' 
    creates regression model with Tensorflow
    input_data: will take the training data and transform it into a tf.float32 placeholder
    '''
    def __init__(self, all_data, learning_rate=1e-3, pca=True, reuse=False, logdir=None):
        self.all_data = all_data
        self.learning_rate = learning_rate
        self.pca = pca
        self.reuse = reuse
        self.LOGDIR = logdir
        
        
    def _format_data(self, label_name='home_win'):
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
            pca = PCA(n_components=30, whiten=True)
            normed_train_data = pca.fit_transform(normed_train_data)
            normed_test_data = pca.transform(normed_test_data)
            normed_valid_data = pca.transform(normed_valid_data)

        self.N_FEATURES = len(normed_train_data[1])
        self.N_SAMPLES = len(normed_train_data)
        
        
    def _create_network(self, n_units=4)
    
        self.x = tf.cast(tf.placeholder(dtype=tf.float32, shape=[None, self.N_FEATURES], name='input'), tf.float32)
        self.y = tf.cast(tf.placeholder(dtype=tf.float32, shape=[None, 1], name='labels'), tf.float32)

        with tf.variable_scope('deep_network') as scope:
            out1 = tf.layers.dense(x, units=n_units, activation=tf.nn.relu)
            _activation_summary(out1)
            out2 = tf.layers.dense(out1, units=n_units, activation=tf.nn.relu)
            _activation_summary(out2)
            out3 = tf.layers.dense(out2, units=n_units, activation=tf.nn.relu)
            _activation_summary(out3)
        
        if self.regression:
            with tf.variable_scope('predictions') as scope:
                preds = tf.layers.dense(out3, units=1, activation=None)
                _activation_summary(preds)
        else: 
            with tf.variable_scope('predictions') as scope:
                preds = tf.layers.dense(out3, units=2, activation=None)
                _activation_summary(preds)
        
        return preds
    
    def _train_regression(self, n_epochs=50, batch_size=32, reuse=False, verbose=True)
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
                    batch_x = train_data[batch*batch_size: min((batch+1)*batch_size, self.N_SAMPLES)]
                    batch_y = train_labels[batch*batch_size: min((batch+1)*batch_size, self.N_SAMPLES)]

                    train_opt = sess.run(optimizer, feed_dict={self.x: batch_x, y: batch_y})
                    train_loss, s = sess.run([loss, summ], feed_dict={self.x: batch_x, y: batch_y})  
                    summary_writer.add_summary(s, batch)


                # all validation samples 
                valid_loss = sess.run(loss, feed_dict={self.x: self.valid_data, self.y: self.valid_labels}) 
                train_losses.append(train_loss)
                valid_losses.append(valid_loss)
                if epoch % 10 == 0 and verbose:
                    print(f'Epoch {epoch} train loss: {train_loss} \nValidation loss: {valid_loss}')
        print(f'\nTraining Complete! Run `tensorboard --logdir={self.LOGDIR+str(lr)}` to see the results.')
                                      
                                      
    def _train_classification(self, n_epochs=50, batch_size=32, reuse=False, verbose=True)
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
            train_losses = []
            train_acc = []
            valid_losses = []
            valid_acc = []
#             test_losses = []
#             test_acc = []

            for epoch in range(1, n_epochs+1):
                for batch in range(self.N_SAMPLES // batch_size): 
                    batch_x = train_data[batch*batch_size: min((batch+1)*batch_size, self.N_SAMPLES)]
                    batch_y = train_labels[batch*batch_size: min((batch+1)*batch_size, self.N_SAMPLES)]

                    train_opt = sess.run(optimizer, feed_dict={self.x: batch_x, y: batch_y})
                    train_loss, s = sess.run([loss, summ], feed_dict={self.x: batch_x, y: batch_y})  
                    summary_writer.add_summary(s, batch)


                # all validation samples 
                valid_loss = sess.run(loss, feed_dict={self.x: self.valid_data, self.y: self.valid_labels}) 
                train_losses.append(train_loss)
                valid_losses.append(valid_loss)
                if epoch % 10 == 0 and verbose:
                    print(f'Epoch {epoch} train loss: {train_loss} \nValidation loss: {valid_loss}')
        print(f'\nTraining Complete! Run `tensorboard --logdir={self.LOGDIR+str(lr)}` to see the results.')
