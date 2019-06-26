import tensorflow as tf

tf.reset_default_graph()

input_data = tf.placeholder(dtype=tf.float32, shape=None)
output_data = tf.placeholder(dtype=tf.float32, shape=None)

f_a = tf.Variable(1, dtype=tf.float32)
f_b = tf.Variable(1, dtype=tf.float32)
f_c = tf.Variable(1, dtype=tf.float32)

model_operation = f_a * input_data * input_data + f_b * input_data + f_c #Equation to regress to data

error = model_operation - output_data
squared_error = tf.square(error)
loss = tf.reduce_mean(squared_error) #loss function (telling how bad it is)

#Function to use to get better (Stochastic Gradient Descent)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.005) #Learning rate is how stochastic it is
train = optimizer.minimize(loss)

init = tf.global_variables_initializer();
x_values = [-2,-1,0,1,2]
y_values = [310,30,-50,70,390]

def carefulRound(x):
    vx = x.eval();
    if (round(vx) - vx) < 0.02:
        return int(round(vx))
    else:
        return vx

with tf.Session() as sess:
    sess.run(init)
    for i in range(2000):
        sess.run(train, feed_dict={input_data:x_values, output_data: y_values})
        if i % 100 == 0:
            print(sess.run([f_a, f_b, f_c]))
    print(sess.run(loss, feed_dict={input_data:x_values, output_data:y_values}))
    print("Final: ", carefulRound(f_a), carefulRound(f_b), carefulRound(f_c))

