import tensorflow
import pandas as pd
tf = tensorflow.compat.v1
tf.disable_eager_execution()
inputDim = 1682
hiddenDim = 1100 #about 2/3 of the input layer
outputDim = 1682


# Parameter specification
X = tf.placeholder(tf.float64,[None,inputDim],name="X") #for input nodes: type,shape, value it will take, reference name

Weight = tf.Variable(tf.random_normal([inputDim,hiddenDim],dtype=tf.float64))
bias = tf.Variable(tf.random_normal([hiddenDim],dtype=tf.float64))
Weight_transpose = tf.transpose(Weight)
bias_prime = tf.Variable(tf.random_normal([inputDim],dtype=tf.float64))

# Creating model
def model(x):
    encoded = tf.nn.sigmoid(tf.matmul(x,Weight)+bias)
    decoded = tf.nn.sigmoid(tf.matmul(encoded,Weight_transpose)+bias_prime)
    return decoded
decoded = model(X)

y_true = X
y_pred = decoded

# Define loss function
loss = tf.losses.mean_squared_error(y_true,y_pred)
optimizer = tf.train.GradientDescentOptimizer(0.02).minimize(loss)

predictions = pd.DataFrame()

# Evaluation metrices

eval_x = tf.placeholder(tf.int32, )
eval_y = tf.placeholder(tf.int32, )
precision, precision_op = tf.metrics.precision(labels=eval_x,predictions=eval_y)

#Initializing variable

init = tf.global_variables_initializer()
local_init = tf.local_variables_initializer()