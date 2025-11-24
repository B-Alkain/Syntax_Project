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
        print(f"TRANSITION COUNTS: \n {transition_counts}")
        print(f"EMISSION COUNTS: \n {emission_counts}")
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
        print(f"TRANSITION PROBS \n {self.transitions}")
        print(f"EMISSION PROBS: \n {self.emissions}")

    def train(self, train_sentences, train_tags):
        transition_counts, emission_counts = self.get_counts(train_sentences, train_tags)
        self.get_probabilities(transition_counts, emission_counts)
    

        



        


