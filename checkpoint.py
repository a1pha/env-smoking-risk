from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf 
import numpy as np
from PIL import Image
from utils import label_map_util
import sys
import IPython.display as display
import matplotlib.pyplot as plt
import os
import random

def main():
	# I.	Create Patient Dictionary
	data_path = '/home/DHE/aj246/PACE_Home_Drive/smoking_data/raw_data' # contains Durham and Pitt Folders

	patient_dict = create_patient_dict(data_path)
	print("Patient Dict Created")

	# II.	Partition into k folds, without patient leakage  
	k = 10
	folds = partition(patient_dict, k) 
	print("Patient Dict Partitioned")

	BATCH_SIZE = 50
	PATIENCE_EPOCHS = 3 
	test_set = folds[0]
	test_imgs = sum(test_set.values(), [])
	test_labels = [int('Smoking' in file_path) for file_path in test_imgs]

	val_set = folds[1]
	val_imgs = sum(val_set.values(), [])
	val_labels = [int('Smoking' in file_path) for file_path in val_imgs]

	train_set = folds
	train_set.remove(folds[0]) 
	train_set.remove(folds[1]) 
	train_imgs = []
	for a in train_set:
		train_imgs = sum(a.values(), train_imgs)
	train_labels = [int('Smoking' in file_path) for file_path in train_imgs]

	num_steps= len(train_imgs) // BATCH_SIZE
	epoch_count = 0
	epoch_losses = []
	prev_loss = 9e99
	patience = -1 # Initialization 

	print("Test, Train, Val splits created")

	# III.	Initialize/Load Models and Define Graph 
	# Object Detector File Paths 

	detection_graph = tf.Graph()
	with detection_graph.as_default():
		od_graph_def = tf.GraphDef()
		with tf.gfile.GFile('./mobilenetssd/frozen_inference_graph.pb', 'rb') as fid:
			serialized_graph = fid.read()
			od_graph_def.ParseFromString(serialized_graph)
			tf.import_graph_def(od_graph_def, name='')

			# Define Input, Output, Label Tensors
			image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
			output_matrix = detection_graph.get_tensor_by_name('concat:0')
			labels = tf.placeholder(tf.float32, [None, 1])

			# Adding operations
			outmat_sq = tf.squeeze(output_matrix)
			logits_max = tf.squeeze(tf.math.reduce_max(outmat_sq, reduction_indices=[0]))
			logits_mean = tf.squeeze(tf.math.reduce_mean(outmat_sq, reduction_indices=[0]))
			logodds = tf.concat([logits_max, logits_mean], 0)
			logodds = tf.expand_dims(logodds, 0)
			logodds.set_shape([None, 1204])

			# Link graphs here 
			# Single Dense Layer Input
			hidden = tf.contrib.layers.fully_connected(inputs=logodds, num_outputs=500, activation_fn=tf.nn.tanh)
			out = tf.contrib.layers.fully_connected(inputs=hidden, num_outputs=1, activation_fn=tf.nn.sigmoid)

			# Define Loss, Training, and Accuracy 
			loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=labels))
			training_step = tf.train.AdamOptimizer(1e-6).minimize(loss, var_list=[hidden, out])
			correct_prediction = tf.equal(tf.round(out), labels)
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		sess = tf.Session(graph=detection_graph)
		while (1):
			accuracy = 0
			total_loss = 0
			epoch_count += 1
			print("Epoch %d" % (epoch_count))
			for i in range(num_steps):
				images, labels = get_batch(train_imgs, train_labels, BATCH_SIZE)
				_, c, acc = sess.run([training_step, loss, accuracy], feed_dict={image_tensor: images, labels: labels})
				print("Batch %d Completed, acc: %f loss: %f" % ((i + 1), acc, c))
				total_loss += c
			if epoch_count == 1:
				saver.save(sess, './saved_models/model')

			# Model not Improving for First Time 
			if ((total_loss >= prev_loss) and patience == -1):
				patience = PATIENCE_EPOCHS

				# Save Model, for later
				saver.save(sess, 'model', global_step=epoch_count,write_meta_graph=False)

			# Model didn't improve over all patience epochs 
			elif ((total_loss >= prev_loss) and patience == 0):
				# Stop Training and Break 
				break
				
			# Model didn't improve over last patience epoch. 
			elif ((total_loss >= prev_loss) and patience > 0):
				# Decrement Patience Counter 
				patience -= 1

			else: 
				# Update best loss seen to last loss
				prev_loss = total_loss
				# Reset Patience 
				patience = -1 




def create_patient_dict(data_path):
	# Creates a dict, key = Patient_ID (Folder Name), element = list of image paths
	patients = {}
	for root, dirs, files in os.walk(data_path):
		for file in files:
			if file.endswith(".jpg") or file.endswith(".JPG"):
				image_path = os.path.join(root, file)
				patient_num = image_path.split('/')[-3]
				if patient_num in patients:
					patients[patient_num].append(image_path)

				else:
					patients[patient_num] = []
					patients[patient_num].append(image_path)

	print(str(len(patients.keys())) + " patients total.")
	return patients

def partition(patient_dict, k):
	partitioned = [] # List of dicts, one for each fold. 
	patients = list(patient_dict.keys())

	num_patients = len(patients)
	p_len = num_patients // k
	remainder = num_patients % k

	random.shuffle(patients)

	for i in range(k):
		if remainder: 
			partitioned.append({k: patient_dict[k] for k in patients[:p_len + 1]})
			patients[p_len + 1:]
			remainder -= 1

		else:
			partitioned.append({k: patient_dict[k] for k in patients[:p_len]})
			patients[p_len:]

	return partitioned



if __name__== "__main__":
		main()
