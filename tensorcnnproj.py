#Nicholas McKillip - working through the lazy programmers class on convolutional neural netwroks
import numpy as np
import tensorflow as tf
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from util import getData, y2indicator, error_rate, init_weight_and_bias, getImageData, init_filter
from sklearn.utils import shuffle

class ConvPoolLayer(object):
	def __init__(self, mi, mo, fw=5, fh=5, poolsz=(2,2)):
		sz = (fw, fh, mi, mo)
		W0 = init_filter(sz, poolsz)
		self.W = tf.Variable(W0)
		b0 = np.zeros(mo, dtype = np.float32)
		self.b = tf.Variable(b0)
		self.poolsz = poolsz
		self.params = [self.W, self.b]

	def forward(self, X):
		conv_out = tf.nn.conv2d(X, self.W, strides = [1, 1, 1, 1], padding = 'SAME')
		conv_out = tf.nn.bias_add(conv_out, self.b)
		pool_out = tf.nn.max_pool(conv_out, ksize=[1,2,2,1], strides=[1,2,2,1],padding = 'SAME')
		return tf.tanh(pool_out)


class HiddenLayer(object):
	def __init__(self, M1, M2, an_id):
		self.id = an_id
		self.M1 = M1
		self.M2 = M2
		W, b = init_weight_and_bias(int(M1), int(M2))
		self.W = tf.Variable(W.astype(np.float32))
		self.b = tf.Variable(b.astype(np.float32))
		self.params = [self.W, self.b]

	def forward(self,X):
		return tf.nn.relu(tf.matmul(X, self.W) + self.b)

class CNN(object):
	def __init__(self, convpool_layer_sizes, hidden_layer_sizes):
		self.convpool_layer_sizes = convpool_layer_sizes
		self.hidden_layer_sizes = hidden_layer_sizes

	def fit(self, X, Y, lr = 10e-4, mu =0.99, decay = 0.999, reg = 10e-3 , epochs = 3, batch_sz = 32, show_fig = True):
		lr = np.float32(lr)
		mu = np.float32(mu)
		reg = np.float32(reg)
		decay = np.float32(decay)
		K = len(set(Y))

		# make a validation set
		X, Y = shuffle(X, Y)
		X = X.astype(np.float32)
		Y = y2indicator(Y).astype(np.float32)
		Xvalid, Yvalid = X[-1000:], Y[-1000:]
		Yvalid_flat = np.argmax(Yvalid, axis = 1)
		X, Y = X[:-1000], Y[:-1000]

		N, d, d, c = X.shape
		mi = c
		outw = d
		outh = d
		self.convpool_layers = []
		for mo, fw, fh in self.convpool_layer_sizes:
			layer = ConvPoolLayer(mi, mo, fw, fh)
			self.convpool_layers.append(layer)
			outw = outw / 2
			outh = outh / 2
			mi = mo
		# intialize hidden layers
		self.hidden_layers = []
		M1 = self.convpool_layer_sizes[-1][0]*outw*outh
		count = 0
		for M2 in self.hidden_layer_sizes:
			h = HiddenLayer(M1, M2, count)
			self.hidden_layers.append(h)
			M1 = M2 #output of last layer is input of next
			count += 1

		# initaliz params of output layers
		W, b = init_weight_and_bias(M1, K)
		self.W = tf.Variable(W, 'W_logreg')
		self.b = tf.Variable(b, 'b_logreg')

		self.params = [self.W, self.b]
		for h in self.convpool_layers:
			self.params += h.params
		for h in self.hidden_layers:
			self.params += h.params

		tfX = tf.placeholder(tf.float32, shape = (None, d, d, c), name = 'X')
		tfY = tf.placeholder(tf.float32, shape = (None, K), name = 'Y')
		act = self.forward(tfX)


		rcost = reg*sum([tf.nn.l2_loss(p) for p in self.params])
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = act, labels =  tfY)) + rcost
		predction = self.predict(tfX)
		train_op = tf.train.RMSPropOptimizer(lr, decay = decay, momentum = mu).minimize(cost)

		n_batches = int(N / batch_sz)
		costs = []
		init = tf.global_variables_initializer()
		with tf.Session() as session:
			session.run(init)
			for i in range(epochs):
				X, Y = shuffle(X, Y)
				for j in range(n_batches):
					Xbatch = X[j*batch_sz:(j*batch_sz + batch_sz)]
					Ybatch = Y[j*batch_sz:(j*batch_sz + batch_sz)]

					session.run(train_op, feed_dict={tfX: Xbatch, tfY: Ybatch})

					if j % 20 == 0:
						c = session.run(cost, feed_dict = {tfX: Xvalid, tfY: Yvalid})
						costs.append(c)

						p = session.run(predction, feed_dict = {tfX: Xvalid, tfY: Yvalid})
						e = error_rate(Yvalid_flat, p)
						print("i:", i, "j:", j, "nb:", n_batches, "cost:", c, "error_rate", e)

		if show_fig:
			plt.plot(costs)
			plt.show()

	def forward(self, X):
		Z = X
		for c in self.convpool_layers:
			Z = c.forward(Z)
		Z_shape = Z.get_shape().as_list()
		Z = tf.reshape(Z,[-1, np.prod(Z_shape[1:])])
		for h in self.hidden_layers:
			Z = h.forward(Z)
		return tf.matmul(Z, self.W) + self.b

	def predict(self, X):
		act = self.forward(X)
		return tf.argmax(act,1)



def main():
	X, Y = getImageData()

	model = CNN(
		convpool_layer_sizes=[(20,5,5), (20,5,5)],
		hidden_layer_sizes = [500, 300],
		)
	model.fit(X,Y, show_fig=True)


if __name__ == '__main__':
	main()
