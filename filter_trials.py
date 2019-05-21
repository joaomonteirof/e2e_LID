import argparse
import numpy as np

def read_scores(path):
	with open(path, 'r') as file:
		utt_labels = file.readlines()

	enroll_list, test_list, scores_list = [], [], []

	for line in utt_labels:
		enroll_utt, test_utt, score = line.split(' ')
		enroll_list.append(enroll_utt)
		test_list.append(test_utt)
		scores_list.append(score.strip())

	return enroll_list, test_list, scores_list

parser = argparse.ArgumentParser(description='Filter scores and leave only confusing languages trials')
parser.add_argument('--scores', type=str, help='Path to complete scores')
parser.add_argument('--out-file', type=str, help='Path to output')
args = parser.parse_args()

enroll_utts, test_utts, scores = read_scores(args.scores)

confusing_languages = ['ct', 'ko', 'zh']

out_data = []

for i in range(len(scores)):
	if enroll_utts[i] in confusing_languages and test_utts[i].split('-')[0] in confusing_languages:
		out_data.append(enroll_utts[i] + ' ' + test_utts[i] + ' ' + scores[i] + '\n')

with open(args.out_file, 'w') as f:
	for item in out_data:
		f.write("%s" % item)

