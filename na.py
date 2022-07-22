import spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp('White dog standing with black and beautiful cat.')

noun_adj_pairs = {}
for chunk in doc.noun_chunks:
    adj = []
    noun = ""
    for tok in chunk:
        if tok.pos_ == "NOUN":
            noun = tok.text
        if tok.pos_ == "ADJ":
            adj.append(tok.text)
    if noun:
        noun_adj_pairs.update({noun:adj})

# expected output
print(noun_adj_pairs)
#{'candy': ['red'], 'book': ['interesting', 'big']} 