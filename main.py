from model.hmm import HMM
import pandas as pd
from sklearn.metrics import classification_report

def get_data(file):
    df = pd.read_csv(file, encoding="latin1")
    sentences = []
    for sentence in df['text'].to_list():
        sentences.append(sentence.split())
    tags = []
    for tag in df['tags'].to_list():
        tags.append(tag.split())
    return sentences, tags

def accuracy(hmm, test_sentences, test_tags):
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
    stats = {}  # tag → [aciertos, total]

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
    # precision, recall, f1 and support for each tag
        # support: The number of occurrences of each label in y_true.
    gold_all = []
    pred_all = []

    for sent, gold in zip(sentences, tags):
        pred = model.viterbi(sent)
        gold_all.extend(gold)
        pred_all.extend(pred)
    
    report = classification_report(gold_all, pred_all, output_dict=True, zero_division=0)
    return report


if __name__ == "__main__":
    # Cargamos train, dev y test igual que en el notebook
    print("---------------LOAD DATA---------------")
    train_sentences, train_tags = get_data("datasets/ud_basque/ud_basque_train.csv")
    dev_sentences, dev_tags     = get_data("datasets/ud_basque/ud_basque_dev.csv")
    test_sentences, test_tags   = get_data("datasets/ud_basque/ud_basque_test.csv")
    
    print(f"# sentences in train: {len(train_sentences)}")
    print(f"# sentences in dev:    {len(dev_sentences)}")
    print(f"# sentences in test:   {len(test_sentences)}")

    # Entrenamos tu HMM con TODO el train
    hmm = HMM()
    hmm.train(train_sentences=train_sentences, train_tags=train_tags)

    # Ejemplo de tagging de una frase del test
    print(f"---------------TAGGING EXAMPLE---------------")
    example_sent = test_sentences[0]
    pred_tags = hmm.viterbi(example_sent)
    print("Tagging example:")
    print(list(zip(example_sent, pred_tags)))

    # # Calculamos accuracy en test
    # acc_test = accuracy(hmm, test_sentences, test_tags)
    # print(f"HMM's accuracy in test: {acc_test:.4f}")    

    ## EVALUATION

    ## PER TAG
    print("---------------REPORT---------------")
    report = evaluate_per_tag(hmm, test_sentences, test_tags)
    print(f"{'':10s}{'Precision':>10s}{'Recall':>10s}{'F-Score':>10s}{'Support':>10s}")

    for k, v in report.items():
        if k == 'accuracy':
            print(f"{k:10s}{v:>10.3f}")
        else:
            print(f"{k:10s}{v['precision']:>10.3f}{v['recall']:>10.3f}{v['f1-score']:>10.3f}{v['support']:>10}")


    print("---------------PER TAG ACCURACY---------------")
    per_tag_accuracy = per_tag_accuracy(hmm, test_sentences, test_tags)
    for tag, acc in per_tag_accuracy.items():
        print(f"{tag}\t{acc:>10.3f}")

    # print(per_tag_accuracy(hmm, test_sentences, test_tags))
    print("---------------ACCURACY---------------")
    print(accuracy(hmm, test_sentences, test_tags))


    