from model.hmm import HMM
import pandas as pd

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


if __name__ == "__main__":
    # Cargamos train, dev y test igual que en el notebook
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
    example_sent = test_sentences[0]
    pred_tags = hmm.viterbi(example_sent)
    print("Tagging example:")
    print(list(zip(example_sent, pred_tags)))

    # Calculamos accuracy en test
    acc_test = accuracy(hmm, test_sentences, test_tags)
    print(f"HMM's accuracy in test: {acc_test:.4f}")    


    