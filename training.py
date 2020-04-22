import tensorflow
import base_model as bm
import user_item_matrix
import numpy as np
import pandas as pd
tf = tensorflow.compat.v1
tf.disable_eager_execution()

input_matrix = user_item_matrix.createMatrix(path='ml-100k/u1.base') #list of numpy arrays
df = pd.read_csv('ml-100k/u1.base',sep='\t',names=['user_id','item_id','rating','timestamp'])
matrix = input_matrix
with tf.Session() as session:
    epochs = 100
    batch_size = 250

    session.run(bm.init) #initializing all variables
    session.run(bm.local_init)
    num_of_batches = int(len(input_matrix)/batch_size)
    input_matrix = np.array_split(input_matrix,num_of_batches) #splitting matrix into no. of batches

    for i in range(epochs):
        avg_cost = 0

        for batch in input_matrix:
            optimization,loss = session.run([bm.optimizer,bm.loss],feed_dict={bm.X:batch})
            avg_cost += loss
        avg_cost /= num_of_batches #average cost of that epoch
        print("Epoch: {}, Loss: {}".format(i+1,avg_cost))
    print("Predictions...")
    input_matrix = np.concatenate(input_matrix,axis=0)

    preds = session.run(bm.decoded,feed_dict={bm.X:input_matrix})
    # print(type(preds)) <class 'numpy.ndarray'>
    # print(len(preds)) 943
    # print(len(preds[0])) #1682
    # print(type(preds[0])) #<class 'numpy.ndarray'>
    # preds is an array of array, each sub-array represesnts a movie, each rating represents predicted rating for
    # that movie by all users
    # preds = pd.DataFrame(preds)
    # print(preds.shape) # (943,1682)
    predictions = bm.predictions # a dataframe
    predictions = predictions.append(pd.DataFrame(preds))
    # print(predictions.shape) #(943,1682)
    predictions = predictions.stack().reset_index(name='rating')
    predictions.columns = ['user_id','item_id','rating']
    #print(predictions)
    matrix = np.array(matrix)
    users = np.arange(1,944)
    items = np.arange(1,1683)

    predictions['user_id'] = predictions['user_id'].map(lambda value: users[value])
    predictions['item_id'] = predictions['item_id'].map(lambda value: items[value])

    #print(predictions)

    keys = ['user_id', 'item_id']
    i1 = predictions.set_index(keys).index
    i2 = df.set_index(keys).index
    k = 50
    recs = predictions[~i1.isin(i2)]
    recs = recs.sort_values(['user_id', 'rating'], ascending=[True, False])
    recs = recs.groupby('user_id').head(k)
    recs.to_csv('recs.tsv', sep='\t', index=False, header=False)

    test = pd.read_csv('ml-100k/u1.test', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'], header=None)
    test = test.drop('timestamp', axis=1)

    test = test.sort_values(['user_id', 'rating'], ascending=[True, False])

    print("Evaluating...")

    p = 0.0
    for user in users[:50]:
        test_list = test[(test.user_id == user)].head(k).as_matrix(columns=['item_id']).flatten()
        recs_list = recs[(recs.user_id == user)].head(k).as_matrix(columns=['item_id']).flatten()

        session.run(bm.precision_op, feed_dict={bm.eval_x: test_list, bm.eval_y: recs_list})

    p = session.run(bm.precision)
    print("Precision@{}: {}".format(k, p))






