from collections import defaultdict, Counter
import numpy as np

class HMM:
    """
    This class creates a Part-of-speech tagger based on a Hidden Markov Model (HMM).
    For that, it implements the viterbi algorithm.
    """
    def __init__(self):
        # Transition probabilities P(tag_i | tag_{i-1})
        self.transitions = defaultdict(lambda: defaultdict(float))
        # Emission probabilities P(word | tag)
        self.emissions = defaultdict(lambda: defaultdict(float))
        # Number of tags
        self.tag_counts = defaultdict(int)
        # Vocabulary
        self.vocab = set()
        # Set of tags
        self.tags = set()
        # Special symbols, start token and unknown
        self.start_token = "*"
        self.unk_token = "UNK"

    def get_counts(self, train_sentences, train_tags):
        """
        This function counts transitions and emissions of the train set. It also updates
        the tag set, the tag counts and the vocabulary (self.tags, self.tag_counts and self.vocab)

        Parameters
        ----------
        train_sentences (list): Sentences of the train set
        train_tags (list): POS tags that correspond to the sentences of the train set

        Returns
        ----------
        transition_counts (dict):   Dictionary where [tag1][tag2] refers to the number of times tag2
                                    follows tag1 in the training data.
        emission_counts (dict):     Dictionary where [tag][word] refers to the number of times a tag
                                    emits a word.
        """
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
        return transition_counts, emission_counts
        
    def get_probabilities(self, transition_counts, emission_counts):
        """
        Converts transition and emission counts into probabilities and updates self.transitions and 
        self.emissions.

        Parameters
        ----------
        transition_counts (dict):   Dictionary where [tag1][tag2] refers to the number of times tag2
                                    follows tag1 in the training data.
        emission_counts (dict):     Dictionary where [tag][word] refers to the number of times a tag
                                    emits a word.
        """
        for prev_tag, following_tag_counts in transition_counts.items():
            sum_transitions = sum(following_tag_counts.values())
            for tag, count in following_tag_counts.items():
                self.transitions[prev_tag][tag] = count / sum_transitions
        for tag, words in emission_counts.items():
            sum_emissions = sum(words.values())
            for word, count in words.items():
                self.emissions[tag][word] = count/sum_emissions

    def train(self, train_sentences, train_tags):
        """
        Trains the HMM by computing transition and emission counts and probabilities.
        It updates self.transitions, self.emissions and tag and vocabulary counts.

        Parameters
        ----------
        train_sentences (list): Sentences of the train set
        train_tags (list): POS tags that correspond to the sentences of the train set
    
        """
        transition_counts, emission_counts = self.get_counts(train_sentences, train_tags)
        self.get_probabilities(transition_counts, emission_counts)
    
    def viterbi(self, sentence):
        """
        This function applies the Viterbi algorithm and returns the most likely sequence of tags
        for the input sentence using the trained HMM.

        Parameters
        ----------
        sentence (list): the input sentence as a list of tokens.

        Returns
        ----------
        best_path (list): the most probable sequence of POS tags corresponding to the input sentence.
        """

        states = list(self.tags)   # Set of possible tags (states of the HMM)
        T = len(sentence) # length of input sentence

        # 1. INITIALIZATION
        V = [{} for _ in range(T)]
        backpointer = [{} for _ in range(T)]

        for tag in states:
            # P(tag | START)
            pi_q = self.transitions[self.start_token].get(tag, 1e-12)
            # P(first_word | tag)
            b_q_o1 = self.emissions[tag].get(sentence[0], 1e-12)

            V[0][tag] = pi_q * b_q_o1
            backpointer[0][tag] = None   # No previous tag at t = 0

        # 2. RECURSION
        for t in range(1, T):
            for tag in states:
                
                # P(current_word | tag)
                emission_prob = self.emissions[tag].get(sentence[t], 1e-12)

                # max over previous states
                max_prob = -1
                best_prev = None

                for prev_tag in states:
                    # P(tag | previous_tag)
                    trans_prob = self.transitions[prev_tag].get(tag, 1e-12)

                    # V[t-1][previous_tag] * P(tag|previous_tag)
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
            # P(STOP | tag)
            stop_prob = self.transitions[tag].get("STOP", 1e-12)
            # V[T-1][tag] * P(STOP | tag)
            prob = V[T-1][tag] * stop_prob

            if prob > max_final_prob:
                max_final_prob = prob
                best_last_tag = tag

        # 4. BACKTRACKING
        best_path = [best_last_tag]

        for t in range(T-1, 0, -1):
            best_path.insert(0, backpointer[t][best_path[0]])

        return best_path
