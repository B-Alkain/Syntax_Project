import csv
import pyconll

# Load CoNLL-U file using pyconll
data = pyconll.load_from_file("../../UD_Basque-BDT/eu_bdt-ud-test.conllu")

# Open output CSV file
with open('../datasets/ud_basque/ud_basque_test.csv', 'w') as file:
    writer = csv.writer(file)

    # Write header row
    writer.writerow(['sentence_id', 'text', 'tags'])

    sentence_count = 0
    for sentence in data:
        id = sentence.id
        upos_tags = []
        words = []
        sentence_count += 1
        for token in sentence:
            # Skip multi-word tokens (del > de el)
            if not token.is_multiword():
                upos_tags.append(token.upos)
                words.append(token.form)

        writer.writerow([
            id,
            " ".join(words),
            " ".join(upos_tags)
        ])

print(f"{sentence_count} Sentences written in the csv.")