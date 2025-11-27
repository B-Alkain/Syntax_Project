from collections import defaultdict, Counter
import numpy as np

class HMM:
    def __init__(self):
        self.transitions = defaultdict(lambda: defaultdict(float))
        self.emissions = defaultdict(lambda: defaultdict(float))
        self.tag_counts = defaultdict(int)
        self.vocab = set()
        self.tags = set()
        self.start_token = "*"
        self.unk_token = "UNK"

    def get_counts(self, train_sentences, train_tags):
        transition_counts = defaultdict(Counter)
        emission_counts = defaultdict(Counter)

        for i, sentence in enumerate(train_sentences):
            prev_tag = self.start_token
            self.tag_counts[prev_tag] += 1

            for j, word in enumerate(sentence):
                current_tag = train_tags[i][j]

                self.vocab.add(word)
                self.tags.add(current_tag)

                transition_counts[prev_tag][current_tag] += 1

                emission_counts[current_tag][word] += 1

                self.tag_counts[current_tag] += 1
                prev_tag = current_tag

            transition_counts[prev_tag]["STOP"] += 1
        # print(f"TRANSITION COUNTS: \n {transition_counts}")
        # print(f"EMISSION COUNTS: \n {emission_counts}")
        return transition_counts, emission_counts
        
    def get_probabilities(self, transition_counts, emission_counts):
        for prev_tag, following_tag_counts in transition_counts.items():
            sum_transitions = sum(following_tag_counts.values())
            for tag, count in following_tag_counts.items():
                self.transitions[prev_tag][tag] = count / sum_transitions
        for tag, words in emission_counts.items():
            sum_emissions = sum(words.values())
            for word, count in words.items():
                self.emissions[tag][word] = count/sum_emissions
        # print(f"TRANSITION PROBS \n {self.transitions}")
        # print(f"EMISSION PROBS: \n {self.emissions}")

    def train(self, train_sentences, train_tags):
        transition_counts, emission_counts = self.get_counts(train_sentences, train_tags)
        self.get_probabilities(transition_counts, emission_counts)
    
    def viterbi(self, sentence):
        states = list(self.tags)   # conjunto de etiquetas
        T = len(sentence)

        # 1. INITIALIZATION
        V = [{} for _ in range(T)]
        backpointer = [{} for _ in range(T)]

        for tag in states:
            pi_q = self.transitions[self.start_token].get(tag, 1e-12)
            b_q_o1 = self.emissions[tag].get(sentence[0], 1e-12)

            V[0][tag] = pi_q * b_q_o1
            backpointer[0][tag] = None   # como en las diapositivas: 0 o None

        # 2. RECURSION
        for t in range(1, T):
            for tag in states:

                emission_prob = self.emissions[tag].get(sentence[t], 1e-12)

                # max over previous states
                max_prob = -1
                best_prev = None

                for prev_tag in states:
                    trans_prob = self.transitions[prev_tag].get(tag, 1e-12)
                    prob = V[t-1][prev_tag] * trans_prob

                    if prob > max_prob:
                        max_prob = prob
                        best_prev = prev_tag

                # viterbi update
                V[t][tag] = max_prob * emission_prob
                backpointer[t][tag] = best_prev

        # 3. TERMINATION (with STOP transition)
        max_final_prob = -1
        best_last_tag = None

        for tag in states:
            stop_prob = self.transitions[tag].get("STOP", 1e-12)
            prob = V[T-1][tag] * stop_prob

            if prob > max_final_prob:
                max_final_prob = prob
                best_last_tag = tag

        # 4. BACKTRACKING
        best_path = [best_last_tag]

        for t in range(T-1, 0, -1):
            best_path.insert(0, backpointer[t][best_path[0]])

        return best_path
