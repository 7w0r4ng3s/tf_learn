import tensorflow as tf

# Class Session
# A class for running TensorFLow operations.
# A Session object encapsulates the environment in which Operation
# objects are executed, and Tensor objects are evaluated.

# Build a graph
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b

# Launch the graph in a session
sess = tf.Session()

# Evaluate the tensor 'c'
print(sess.run(c))


# Use tf.Session.close method to close the session when they are 
# no longer required
# OR use the session as a context manager

# Using the 'close()' method
sess.close()

# Using the context manager
with tf.Session() as sess:
	sess.run(c)

# The ConfigProto buffer exposes various configuration options for a 
# session
# To create a session that uses soft constraints for device placement,
# and log the resulting placement decisions, create a session as follows

# Launch a graph in a session that allows soft device placement and
# logs the placement decisions
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
										log_device_placement=True))

