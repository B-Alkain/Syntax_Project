from model.hmm import HMM
import pandas as pd

def get_data(file):
    df = pd.read_csv(file)
    sentences = []
    for sentence in df['text'].to_list():
        sentences.append(sentence.split())
    tags = []
    for tag in df['tags'].to_list():
        tags.append(tag.split())
    return sentences, tags

if __name__ == "__main__":
    train_sentences, train_tags = get_data("datasets/ud_basque/ud_basque_train.csv")
    # PROBA
    sents = []
    sents.append(train_sentences[0])
    sents.append(train_sentences[2])    
    
    tags = []
    tags.append(train_tags[0])
    tags.append(train_tags[2])
    for sent, tag in zip(sents, tags):
        print(f"SENTECE: {sent}, \n TAG: {tag} \n") 
    hmm = HMM()
    hmm.train(train_sentences=sents, train_tags=tags)

    test_sentence = train_sentences[0]  # ejemplo
    predicted = hmm.viterbi(test_sentence)
    print(predicted)
    # print(train_sentences[0])
    # print(train_tags[0])

    