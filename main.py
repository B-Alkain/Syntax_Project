from model.hmm import HMM
import pandas as pd
from sklearn.metrics import classification_report

def get_data(file):
    """
    Loads a CSV file that contains the colums 'text' and 'tags' and converts them
    into lists of lists.

    Parameters
    ----------
    file (str): Path to the CSV file. The file must contain two columns:
        - 'text': sentences as space-separated tokens
        - 'tags': corresponding POS tags as space-separated labels

    Returns
    ----------
    sentences (list): list of sentences, where each sentence is a list of words
    tags (list): list of tag sequences, where each list corresponds to a sentence
    """
    df = pd.read_csv(file, encoding="latin1")
    sentences = []
    for sentence in df['text'].to_list():
        sentences.append(sentence.split())
    tags = []
    for tag in df['tags'].to_list():
        tags.append(tag.split())
    return sentences, tags

def accuracy(hmm, test_sentences, test_tags):
    """
    Computes tagging accuracy of a trained HMM on a test dataset.

    Parameters
    ----------
    hmm (HMM): A trained Hidden Markov Model.
    test_sentences (list): Sentences in the test set as lists of words
    test_tags (list): POS tag sequences corresponding to test_sentences

    Returns
    ----------
    accuracy (float): overall tagging accuracy
    """
    total = 0
    correct = 0
    
    for words, gold_tags in zip(test_sentences, test_tags):
        pred_tags = hmm.viterbi(words)
        for p, g in zip(pred_tags, gold_tags):
            total += 1
            if p == g:
                correct += 1
    
    return correct / total if total > 0 else 0.0

def per_tag_accuracy(model, sentences, tags):
    """
    Computes per-tag accuracy for a trained HMM model.
    
    Parameters
    ----------
    model (HMM): A trained Hidden Markov Model.
    sentences (list): Sentences in the test set as lists of words.
    test_tags (list): POS tag sequences corresponding to test_sentences

    Returns
    ----------
    tag_acc (dict): A dictionary containing each tag and its accuracy.
    """
    stats = {}  # tag → [correct_predictions, total]

    for sent, gold in zip(sentences, tags):
        pred = model.viterbi(sent)
        for p, g in zip(pred, gold):
            if g not in stats:
                stats[g] = [0,0]
            if p == g:
                stats[g][0] += 1
            stats[g][1] += 1

    tag_acc = {tag: correct/total for tag, (correct,total) in stats.items()}
    return tag_acc

def evaluate_per_tag(model, sentences, tags):
    """
    Computes precision, recall, F1-score and support for each POS tag.

    Parameters
    ----------
    model (HMM): A trained Hidden Markov Model.
    sentences (list): Sentences in the test set as lists of words.
    tags (list): POS tag sequences corresponding to test_sentences

    Returns
    ----------
    A dictionary containing per-tag precision, recall, F1-score and support (number of 
    occurrences), as well as accuracy, macro and weighted averages.
    """
    gold_all = []
    pred_all = []

    for sent, gold in zip(sentences, tags):
        pred = model.viterbi(sent)
        gold_all.extend(gold)
        pred_all.extend(pred)
    
    report = classification_report(gold_all, pred_all, output_dict=True, zero_division=0)
    return report


if __name__ == "__main__":
    # ---------------------------------------------
    # 1. LOAD DATA
    # ---------------------------------------------
    print("---------------LOAD DATA---------------")
    train_sentences, train_tags = get_data("datasets/ud_catalan/ud_catalan_train.csv")
    dev_sentences, dev_tags     = get_data("datasets/ud_catalan/ud_catalan_dev.csv")
    test_sentences, test_tags   = get_data("datasets/ud_catalan/ud_catalan_test.csv")
    
    print(f"# sentences in train: {len(train_sentences)}")
    print(f"# sentences in dev:    {len(dev_sentences)}")
    print(f"# sentences in test:   {len(test_sentences)}")

    # ---------------------------------------------
    # 2. TRAIN THE HMM ON THE FULL TRAIN SET
    # ---------------------------------------------
    hmm = HMM()
    hmm.train(train_sentences=train_sentences, train_tags=train_tags)

    # ---------------------------------------------
    # 3. TAGGING EXAMPLE
    # ---------------------------------------------

    print(f"---------------TAGGING EXAMPLE---------------")
    example_sent = test_sentences[0]
    pred_tags = hmm.viterbi(example_sent)
    print("Tagging example:")
    print(list(zip(example_sent, pred_tags))) 

    # ---------------------------------------------
    # 4. EVALUATION
    # ---------------------------------------------

    print("---------------REPORT---------------")
    report = evaluate_per_tag(hmm, test_sentences, test_tags)
    print(f"{'':10s}{'Precision':>10s}{'Recall':>10s}{'F-Score':>10s}{'Support':>10s}")

    for k, v in report.items():
        if k == 'accuracy':
            print(f"{k:10s}{v:>10.3f}")
        else:
            print(f"{k:10s}{v['precision']:>10.3f}{v['recall']:>10.3f}{v['f1-score']:>10.3f}{v['support']:>10}")

    # ---------------------------------------------
    # 5. PER-TAG ACCURACY
    # ---------------------------------------------
    print("---------------PER TAG ACCURACY---------------")
    per_tag_accuracy = per_tag_accuracy(hmm, test_sentences, test_tags)
    for tag, acc in per_tag_accuracy.items():
        print(f"{tag}\t{acc:>10.3f}")

    # ---------------------------------------------
    # 6. GLOBAL ACCURACY
    # ---------------------------------------------

    # print(per_tag_accuracy(hmm, test_sentences, test_tags))
    print("---------------ACCURACY---------------")
    print(accuracy(hmm, test_sentences, test_tags))


    