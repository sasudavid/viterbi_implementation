
#WRITTEN BY: DAVID SASU

import codecs
import math
import torch
from collections import OrderedDict

'''
preprecess_train_data is used to extract the data from the training data for training

Params:
	- file1: The name of the training file
	- percentage_of_training_data: The percentage of the data to use for training

Returns:
	- words_from_train: A list contaning the words extracted from the traning data
	- pos_words_dict: A dictionary containing the parts of speech as the keys and the words corresponding to the 
					various parts of speech as values.
	-unique_pos_from_train: A list containing the unique parts of speech that are represented in the training data
	-pos_pos_dict: A dictionary containing the parts of speech in the training data as keys and the parts of speech that
					appear before them in the data as values.
'''

def preprocess_train_data(file1, percentage_of_training_data):
	inputfile = open(file1, "r")
	ignored = ['PUNCT', 'SYM','X','_']
	bad_pos = ['NUM','PROPN']
	pos_of_a_sentence = []
	pos_of_all_sentences_without_start = []
	pos_of_all_sentences_with_start = []
	words_from_train = []
	unique_pos_from_train = []
	pos_words_dict = {}
	pos_pos_dict = {}
	num_of_train_sentences = 0
	num_of_sentences_to_use = math.ceil((percentage_of_training_data/100) * 5663)
	for line in inputfile:
		if num_of_train_sentences == num_of_sentences_to_use:
			break
		if "# text =" in line:
			if len(pos_of_a_sentence) != 0:
				pos_of_all_sentences_without_start.append(pos_of_a_sentence)
				pos_of_a_sentence = []
				num_of_train_sentences += 1
		else:
			split_line = line.split('\t')
			try:
				num_line = int(split_line[0])
				word = split_line[1].lower()
				pos = split_line[3]


				if pos in bad_pos:
					word = "<unk>"

				if pos not in ignored:
					unique_pos_from_train.append(pos)
					pos_of_a_sentence.append(pos)
					words_from_train.append(word)
					
				if pos not in pos_words_dict:
					pos_words_dict[pos] = ""+word
				else:
					pos_words_dict[pos] = pos_words_dict[pos] +","+word
			except:
				continue

	for pos in pos_words_dict:
		pos_words_dict[pos] = pos_words_dict[pos].split(',')


	unique_pos_from_train = list(set(unique_pos_from_train))
	unique_pos_from_train.sort()
	unique_pos_from_train = ['<s>']+unique_pos_from_train
	words_from_train = list(set(words_from_train))
	words_from_train.sort()
	for sentence_pos_values in pos_of_all_sentences_without_start:
		pos_of_all_sentences_with_start.append(['<s>']+sentence_pos_values)
	for i in unique_pos_from_train:
		for j in pos_of_all_sentences_with_start:
			for k in range(1, len(j)):
				if j[k] == i:
					if i not in pos_pos_dict:
						pos_pos_dict[i] = ""+j[k - 1]
					else:
						pos_pos_dict[i] = pos_pos_dict[i]+","+j[k - 1]
	for pos in pos_pos_dict:
		pos_pos_dict[pos] = pos_pos_dict[pos].split(',')
	pos_words_dict = OrderedDict(sorted(pos_words_dict.items()))
	pos_pos_dict = OrderedDict(sorted(pos_pos_dict.items()))
	print("percentage of training data being used for training: "+str(percentage_of_training_data)+"%")


	return words_from_train, pos_words_dict, unique_pos_from_train, pos_pos_dict


'''
build_emission_probability_matrix is used to construct the emission probability matrix that contains the words
in the training data as rows of the matrix and their corresponding parts of speech as columns of the matrix.

Params:
	- vocabulary: A list of all the words from the training data
	- emission_dict: A dictionary containing the parts of speech as keys and the words that belong to those parts of
					speech as values
Returns:
	- words_from_train: A list of the words from the train data to function as the rows of the emission probability matrix 
	- pos_emission_dict_keys: A list of the parts of speech from the train data to function as the columns of the emission probability matrix
	- torch.tensor(probability_matrix): A tensor representing the emission probability matrix
'''
def build_emission_probability_matrix(vocabulary, emission_dict):
	pos_emission_dict_keys = list(emission_dict.keys())
	words_from_train = vocabulary
	probability_matrix = []
	for word in words_from_train:
		word_pos_probabilities_list = []
		count = 0
		for pos in emission_dict:
			for element in emission_dict[pos]:
				if element == word:
					count += 1
			prob_of_word_given_pos = (count/len(emission_dict[pos]))
			word_pos_probabilities_list.append(prob_of_word_given_pos)
			count = 0
		probability_matrix.append(word_pos_probabilities_list)




	return words_from_train, pos_emission_dict_keys, torch.tensor(probability_matrix)




	


'''
build_transition_probability_matrix is used to construct the transition probability matrix that contains
parts of speech as rows and the parts of speech that they transition to as columns.

Params:
	- states: A list of parts of speech in the training data
	- transition_dict: A dictionary containing the parts of speech in the training data as keys and their corresponding
					transition parts of speech as values.

Returns:
	- pos_transition_dict_keys: A list of parts of speech from the training data to function as the rows of the transition probability matrix
	- unique_pos_from_train: A list of parts of speech from the training data to function as the columns of the transition probability matrix
	- torch.tensor(probability_matrix): A tensor representing the transition probability matrix
'''
def build_transition_probability_matrix(states, transition_dict):
	pos_transition_dict_keys = list(transition_dict.keys())
	unique_pos_from_train = states
	probability_matrix = []
	alpha = 0.25
	for dict_entry in transition_dict:
		row_probability_values = []
		for unique_pos in unique_pos_from_train:
			count = 0
			for pos_entry in transition_dict[dict_entry]:
				if pos_entry == unique_pos:
					count += 1

			transition_prob_of_pos = count / len(transition_dict[dict_entry]) + alpha
			row_probability_values.append(transition_prob_of_pos)
		probability_matrix.append(row_probability_values)

		

	return pos_transition_dict_keys, unique_pos_from_train, torch.tensor(probability_matrix)


'''
retrieve_word_emission_probabilities is used to retrieve the probabilities of a word with respect to the different parts of speech from the emission matrix

Params:
	- word: The word whose emission probabilities are to be retrieved
	- emission_matrix_rows: All the words in the training data representing the emission matrix rows
	- emission_matrix_columns: All the parts of speech in the training data representing the emission matrix columns
	- emission_matrix: A matrix representing the the emission probabilities of all the words in the train data

Returns: 
	- emission_matrix[word_index,:]: A list representing the emission probabilities of the word provided as a parameter to the function
'''
def retrieve_word_emission_probabilities(word, emission_matrix_rows, emission_matrix_columns, emission_matrix):
	if word in emission_matrix_rows:
		word_index = emission_matrix_rows.index(word)
	else:
		word_index = emission_matrix_rows.index("<unk>")

	return emission_matrix[word_index,:]


'''
compute_viterbi is used to apply the viterbi algorithm to compute the most likely parts of speech tag sequence given a test sentence

Params:
	- sentence_list: A list containing the words in the test sentence
	- transition_matrix: A matrix containing the probabilities of transitioning from one state to another
	- transition_matrix_columns: A list containing the states which represent the columns of the transition matrix
	- transition_matrix_rows: A list containing the states which represent the rows of the transition matrix
	- emission_matrix: A matrix containing the probabilities that a state produced a particular word
	- emission_matrix_columns: A list containing the states which represent the columns of the emission matrix
	- emission_matrix_rows: A list containing the words which represent the rows of the emission matrix

Returns:
	- pos_tags_for_sentence: A list representing the predicted parts of speech tags for the given test sentence.
'''
def compute_viterbi(sentence_list, transition_matrix, transition_matrix_columns, transition_matrix_rows, emission_matrix, emission_matrix_columns, emission_matrix_rows):
	viterbi_matrix = torch.zeros(len(sentence_list), len(transition_matrix_columns)-1)
	s_start_transition_probabilities = transition_matrix[:,0]
	viterbi_paths_dict = {}
	first_word_row = retrieve_word_emission_probabilities(sentence_list[0], emission_matrix_rows, emission_matrix_columns, emission_matrix)
	viterbi_matrix[0,:] = first_word_row * s_start_transition_probabilities
	pos_tags_for_sentence = []


	if len(viterbi_matrix) > 1:
		for viterbi_matrix_row_index in range(1, len(viterbi_matrix)):
			transition_matrix_row_index = 0
			for viterbi_matrix_column_index in range(len(viterbi_matrix[viterbi_matrix_row_index, :])):
				transition_vector = transition_matrix[transition_matrix_row_index, 1:] * viterbi_matrix[viterbi_matrix_row_index-1, :] 
				maximum_value_in_transition_vector = max(transition_vector)
				index_of_max = transition_vector.tolist().index(maximum_value_in_transition_vector) + 1
				viterbi_matrix[viterbi_matrix_row_index, viterbi_matrix_column_index] = maximum_value_in_transition_vector
				if viterbi_matrix_column_index in viterbi_paths_dict:
					viterbi_paths_dict[viterbi_matrix_column_index] = viterbi_paths_dict[viterbi_matrix_column_index] + ","+ str(index_of_max)
				else:
					viterbi_paths_dict[viterbi_matrix_column_index] = ""+str(index_of_max)
				transition_matrix_row_index += 1
			viterbi_matrix[viterbi_matrix_row_index,:] = viterbi_matrix[viterbi_matrix_row_index,:] * retrieve_word_emission_probabilities(sentence_list[viterbi_matrix_row_index], emission_matrix_rows, emission_matrix_columns, emission_matrix)

		largest_total_probability = max(viterbi_matrix[len(viterbi_matrix)-1, :])
		index_of_largest_probability = viterbi_matrix[len(viterbi_matrix)-1, :].tolist().index(largest_total_probability)
		best_path = (viterbi_paths_dict[index_of_largest_probability]+','+str(index_of_largest_probability)).split(',')

	


		for index in best_path:
			num_index = int(index)
			pos_tags_for_sentence.append(transition_matrix_columns[num_index])


	else:
		maximum_value = max(viterbi_matrix[0,:])
		index_of_max = viterbi_matrix[0,:].tolist().index(maximum_value)
		pos_tags_for_sentence.append(transition_matrix_columns[index_of_max])



	return pos_tags_for_sentence


'''
preprocess_test_data is used to preprocess the test data set.

Params:
	- filename: A string representing the name of the file containing the test data

Returns:
	- test_sentences: A list containing the test sentences obtained from the test data set
	- pos_of_test_sentences: A list containing the parts of speech predicted for each of the test sentences in the test data set.
'''

def preprocess_test_data(filename):
	inputfile = open(filename, "r")
	ignored = ['PUNCT', 'SYM','X','_']
	test_sentences = []
	test_sentence = []
	pos_of_test_sentences = []
	pos_of_one_test_sentence = []

	for line in inputfile:
		if "# text =" in line:
			if len(pos_of_one_test_sentence) != 0:
				pos_of_test_sentences.append(pos_of_one_test_sentence)
				test_sentences.append(test_sentence)
				pos_of_one_test_sentence = []
				test_sentence = []
		else:
			split_line = line.split('\t')
			try:
				num_line = int(split_line[0])
				
				if split_line[3] not in ignored:
					test_sentence.append(split_line[1].lower())
					pos_of_one_test_sentence.append(split_line[3])

			except:
				continue

	return test_sentences, pos_of_test_sentences

'''
compute_accuracy is used to compute the accuracy of the parts of speech tagger

Params:
	- sentences: A list of sentences from the test data to evaluate the parts of speech tagger
	- pos_tags: A list containing the parts of speech tags of the test sentences
	- transition_matrix: A matrix containing the probabilities of transitioning from one state to another
	- transition_matrix_columns: A list containing the states which represent the columns of the transition matrix
	- transition_matrix_rows: A list containing the states which represent the rows of the transition matrix
	- emission_matrix: A matrix containing the probabilities that a state produced a particular word
	- emission_matrix_columns: A list containing the states which represent the columns of the emission matrix
	- emission_matrix_rows: A list containing the words which represent the rows of the emission matrix

Returns:
	- average_sentence_score: The average accuracy score for the test sentences
	- average_pos_token_score: The average accuracy score for the parts of speech tags of the test sentences
'''
def compute_accuracy(file,sentences, pos_tags, transition_matrix, transition_matrix_columns, transition_matrix_rows, emission_matrix, emission_matrix_columns, emission_matrix_rows):
	outputfile = open(file, 'a')
	token_score = 0
	complete_sentence_score = 0
	num_tokens_tested = 0
	for i in range(len(sentences)):
		predicted_pos_tags = compute_viterbi(sentences[i], transition_matrix, transition_matrix_columns, transition_matrix_rows, emission_matrix, emission_matrix_columns, emission_matrix_rows)
		outputfile.write("sentence: "+str(sentences[i])+'\n'+"sentence_parts_of_speech: "+str(pos_tags[i])+'\n'+", predicted_parts_of_speech: "+str(predicted_pos_tags)+'\n')

		if predicted_pos_tags == pos_tags[i]:
			complete_sentence_score += 1

		for j in range(len(predicted_pos_tags)):
			num_tokens_tested += 1
			if predicted_pos_tags[j] == pos_tags[i][j]:
				token_score += 1



	average_sentence_score = round((complete_sentence_score / len(sentences)) * 100, 3)

	average_pos_token_score = round((token_score / num_tokens_tested) * 100, 3)
	outputfile.close()

	return average_sentence_score, average_pos_token_score













if __name__ == "__main__":
	print("Running...")
	for i in range(10,110,10):
		outputfile = "test_predicted_pos_"+str(i)+".txt"
		words, words_pos, pos, pos_pos = preprocess_train_data("en_gum-ud-train.conllu", i)
		generated_emission_matrix_rows, generated_emission_matrix_columns, generated_emission_matrix = build_emission_probability_matrix(words, words_pos)
		generated_transition_matrix_rows, generated_transition_matrix_columns, generated_transition_matrix = build_transition_probability_matrix(pos, pos_pos)
		generated_test_sentences, generated_pos_of_test_sentences = preprocess_test_data("en_gum-ud-test.conllu")
		sentence_accuracy_score, pos_accuracy_score = compute_accuracy(outputfile,generated_test_sentences, generated_pos_of_test_sentences, generated_transition_matrix, generated_transition_matrix_columns, generated_transition_matrix_rows, generated_emission_matrix, generated_emission_matrix_columns, generated_emission_matrix_rows)
		print("sentence accuracy: ", sentence_accuracy_score)
		print("pos accuracy: ", pos_accuracy_score)













