import spacy

nlp = spacy.load('en_core_web_sm')
doc = nlp('The small kitchen is beautiful and clean and the fruit is fresh')

noun_adj = {}
adj = []
for tok in doc:
    if tok.pos_ == "NOUN":
        noun_adj[tok.text] = []
    if tok.pos_ == "ADJ":
        adj.append(tok.text)


def get_noun_dependency(token):
    if token.pos_ == "NOUN":
        print(token.text)
        return token.text

    if token.pos_ == "AUX":
        # Check the noun in children (only 1 level)
        children = [i.text for i in token.children]
        for noun in noun_adj:
            if noun in children:
                return noun
        return ""

    return get_noun_dependency(token.head)


for tok in doc:
    # direct dependency
    if tok.pos_ == "ADJ":
        noun = get_noun_dependency(tok)
        if noun in noun_adj.keys():
            noun_adj[noun].append(tok.text)

print(noun_adj)
