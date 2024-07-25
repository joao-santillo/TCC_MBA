import tensorflow as tf

# Cria um tensor e realiza uma operação
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
c = tf.matmul(a, b)

print(c)

# Verifica em qual dispositivo a operação foi realizada
tf.debugging.set_log_device_placement(True)

with tf.device('/GPU:0'):
    c = tf.matmul(a, b)
    print(c)