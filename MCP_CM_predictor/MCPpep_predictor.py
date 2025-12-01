import os, sys
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

checkpoint_dir = './checkpoint_MCP'
result_dir = './result'

n_hidden = 200
keep_prob = 0.7

def conv2d(x, W, b):
    x = tf.reshape(x, shape = [-1, 700, 44])
    x = tf.nn.conv1d(x, W, 1, padding = 'SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def multi_conv(x, weights, biases):
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv2 = conv2d(x, weights['wc2'], biases['bc2'])
    conv3 = conv2d(x, weights['wc3'], biases['bc3'])
    convs = tf.concat([conv1, conv2, conv3], 2)
    return convs

def BiRNN(x, weights, biases, seq_length):
    stacked_fw_rnn = []
    stacked_bw_rnn = []
    for i in range(3):
        lstm_fw_cell = tf.contrib.rnn.GRUCell(n_hidden)
        lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, input_keep_prob = 1.0, output_keep_prob = keep_prob)
        stacked_fw_rnn.append(lstm_fw_cell)
        lstm_bw_cell = tf.contrib.rnn.GRUCell(n_hidden)
        lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, input_keep_prob = 1.0, output_keep_prob = keep_prob)
        stacked_bw_rnn.append(lstm_bw_cell)
    mcell_fw = tf.contrib.rnn.MultiRNNCell(stacked_fw_rnn, state_is_tuple = True) 
    mcell_bw = tf.contrib.rnn.MultiRNNCell(stacked_bw_rnn, state_is_tuple = True) 
    init_state_fw = mcell_fw.zero_state(1, dtype = tf.float32)
    init_state_bw = mcell_bw.zero_state(1, dtype = tf.float32)
    output, states = tf.nn.bidirectional_dynamic_rnn(mcell_fw, mcell_bw, x, sequence_length = seq_length, time_major = False, initial_state_fw = init_state_fw, initial_state_bw = init_state_bw, dtype = tf.float32)
    lstmoutputs = tf.concat(output, 2)
    lstmoutputs = tf.concat([lstmoutputs, x], 2)   
    outputs = tf.reshape(lstmoutputs, [-1, 2 * n_hidden + 192])
    return tf.matmul((tf.matmul(outputs, weights['out']) + biases['out']), weights['out2']) + biases['out2']

weights = {
    'wc1': tf.Variable(tf.truncated_normal([3, 44, 64])),
    'wc2': tf.Variable(tf.truncated_normal([7, 44, 64])),
    'wc3': tf.Variable(tf.truncated_normal([11, 44, 64])),
    'out': tf.Variable(tf.truncated_normal([ 2 * n_hidden  + 192, 1])),
    'out2': tf.Variable(tf.truncated_normal([1, 1]))
}

biases = {
    'bc1': tf.Variable(tf.truncated_normal([64])),
    'bc2': tf.Variable(tf.truncated_normal([64])),
    'bc3': tf.Variable(tf.truncated_normal([64])),
    'out': tf.Variable(tf.truncated_normal([1])),
    'out2': tf.Variable(tf.truncated_normal([1]))
}

def fasta_load(fasta_dir, tail = '.fasta'):
    fp = open(fasta_dir + tail, 'r')
    lines = fp.readlines()
    fp.close()
    sequence = ''
    for line in lines[1:]:
        sequence = sequence + line.split()[0]
    return sequence

def feature_load(feature_dir, start = 0, end = 23, tail = '.i1c'):
    fp = open(feature_dir + tail, 'r')
    lines = fp.readlines()
    fp.close()
    feature = []
    for line in lines[1:]:
        temp = line.split()
        if len(temp) == 0:
            continue  
        feature.append(temp[start:end])
    return np.array(feature, dtype = np.float32)

def sort_len(seq_info):
    length_subs = []
    for i, data in enumerate(seq_info):
        length = len(data[b'sequence'])
        length_sub = [length, i, data[b'name'].decode('utf-8')]
        length_subs.append(length_sub)
    return sorted(length_subs, key = lambda seq_length: seq_length[0], reverse = True)

def pkl_read(pkl_file, encoding = 'bytes'):
    fp = open(pkl_file, 'rb')
    contents = pickle.load(fp, encoding = encoding)
    fp.close()
    return contents

def pkl_save(pkl_data, save_dir):
    fp = open(save_dir, 'wb')
    pickle.dump(pkl_data, fp)
    fp.close()
    return print('Data saved in file: ' + save_dir)

def files2pkl(id_list, pkl_dir, pkl_name, fasta_dir, feature_dir):
    pkl = []
    report_i = 0
    for pdbid in id_list:
        sequence = fasta_load(fasta_dir + pdbid)
        seq_feature = feature_load(feature_dir + pdbid, start = 0, end = 23, tail = '.i1c')
        pkl_dict = {b'name': pdbid.encode('utf-8'),
                    b'sequence': sequence.encode('utf-8'),
                    b'seq_feature': seq_feature}
        pkl.append(pkl_dict)
    return pkl_save(pkl, pkl_dir + pkl_name)

def sub2train_fl(length_subs, seq_infos, out_length = 700):
    max_length = 0
    seq_ins = []
    for length, sub, _ in length_subs:
        if length > max_length:
            max_length = length
        pad_length = out_length - length
        seq_data = seq_infos[sub]
        sequence = seq_data[b'sequence'] 
        seq_feature = seq_data[b'seq_feature']
        sequence = str(sequence, encoding = "utf8")
        one_hot = np.zeros((len(sequence),21),dtype = np.int)
        aas = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','X']
        for i, aa in enumerate(sequence):
            if aa not in aas:
                aa = 'X'
                one_hot[i][aas.index(aa)] += 1
        seq_in = np.concatenate([one_hot, seq_feature], axis = 1)
        seq_ins.append(np.pad(seq_in, ((0, pad_length), (0, 0)), 'constant'))

    return sequence, np.array(seq_ins)[:, :, :]

def modeling(test_set, training_steps = 0, batch_size = 1, train_rate_initial = 0.001):

	with tf.Session() as sess:
		x = tf.placeholder(tf.float32, [batch_size, 700, 44])
		y = tf.placeholder(tf.float32, [batch_size, 700, 1])
		seq_length = tf.placeholder("int32", [None])
		mask_weight = tf.placeholder(tf.float32, [None, 700])
		mask = tf.reshape(mask_weight, [-1])		
		x_convs = multi_conv(x, weights, biases)
		pred = BiRNN(x_convs, weights, biases, seq_length)
		global_step = training_steps
		pred = tf.nn.sigmoid(pred)		
		train_rate = tf.train.exponential_decay(train_rate_initial, global_step=training_steps, decay_steps=10, decay_rate=0.99)		

		g_list = tf.global_variables()
		cross_entropy = tf.reduce_sum(tf.pow((pred-y),2))
		l2norm = 0.0001*tf.reduce_sum([ tf.nn.l2_loss(v) for v in g_list ]) 
		mean_loss = cross_entropy + l2norm
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			train_step = tf.train.AdamOptimizer(train_rate).minimize(mean_loss) 
		train_grad = tf.gradients(pred, x)
		saver = tf.train.Saver(var_list = g_list, max_to_keep = 1000)
		with sess.as_default():
			seq_infos_valid = pkl_read(test_set)
			length_subs_valid = sort_len(seq_infos_valid)
			one_train_step_valid = len(length_subs_valid) // batch_size
			tf.get_variable_scope()
			if os.path.exists(checkpoint_dir):
				modeldir = os.path.join(checkpoint_dir, 'Epoch')
				if os.path.exists(modeldir): 
					try:
						saver.restore(sess, os.path.join(modeldir, 'model.ckpt'))
						print('Loading the checkpoint: ' + os.path.join(modeldir, 'model.ckpt'))
					except:
						print('The check point directory does not exist.\nTest failed.')
						return 0
				else:
					print('The check point directory does not exist.\nTest failed.')
					return 0
			else:
				os.makedirs(checkpoint_dir)
				tf.global_variables_initializer().run()
				print('This is a new training.\nInitializing new parameters.')

			if not os.path.exists(result_dir):
 				os.makedirs(result_dir)

			for i in range(one_train_step_valid):
				sequence, train_in = sub2train_fl(length_subs_valid[i * batch_size:(i + 1) * batch_size], seq_infos_valid, out_length = 700)
				seq_len = [1] * batch_size					
				mask = np.ones((batch_size, 700))
				for f,length in enumerate(seq_len):
					mask[f, length:] = 0	
				mask_wei = mask
				test_out = sess.run(pred, feed_dict = {x: train_in[:,:,:44], seq_length: seq_len, mask_weight: mask})
				pred_out = test_out[0:length_subs_valid[i][0]]

				fp = open(os.path.join(result_dir, length_subs_valid[i][2] + '_MCP.txt'), 'a')
				fp.write('%s %s %s\n' % ('#', 'AA', 'MCP'))
				for j in range(length_subs_valid[i][0]):
					fp.write('%d %s %g\n' % (j + 1 ,sequence[j], pred_out[j]))
				fp.close()				
	return 0

list1 = sys.argv[2]

id_list = list1.split()

files2pkl(id_list, sys.argv[1] +'/', 'MCP-'+str(len(id_list)) + '.pkl', sys.argv[1] + '/', sys.argv[1] + '/')

modeling(sys.argv[1] + '/' + 'MCP-'+ str(len(id_list)) + '.pkl')
