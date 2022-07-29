import json

import pandas as pd
import spacy

nlp = spacy.load('en_core_web_sm')
# doc = "A clean bathroom"


def print_pos(str):
    doc = nlp(str)
    co_dict = {
        "token.text": [],
        "token.lemma_": [],
        "token.pos_": [],
        "token.tag_": [],
        "token.dep_": [],
        "token.shape_": [],
        "token.is_alpha": [],
        "token.is_stop": [],
        "token.head": [],
    }

    for token in doc:
        # print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
        #       token.shape_, token.is_alpha, token.is_stop)
        co_dict["token.text"].append(token.text)
        co_dict["token.lemma_"].append(token.lemma_)
        co_dict["token.pos_"].append(token.pos_)
        co_dict["token.tag_"].append(token.tag_)
        co_dict["token.dep_"].append(token.dep_)
        co_dict["token.shape_"].append(token.shape_)
        co_dict["token.is_alpha"].append(token.is_alpha)
        co_dict["token.is_stop"].append(token.is_stop)
        co_dict["token.head"].append(token.head)

        # if token.pos_ == "NOUN":
        #     pos_dict["NOUN"].append(token.text)
        #     pos_dict["LEMMA"].append(token.lemma_)

    df = pd.DataFrame(data=co_dict)
    print(df)
    # print(json.dumps(pos_dict, indent=4))


def get_noun_dependency(token, noun_adj, prev_token_text=""):
    prev_token_text = token.text
    if token.pos_ == "NOUN":
        return token.text

    if token.pos_ == "AUX" or token.pos_ == "VERB":
        if token.head.pos_ == "NOUN":
            return token.head.text
        # Check the noun in children (only 1 level)
        children = [i.text for i in token.children]
        for noun in noun_adj:
            if noun in children:
                return noun
        if token.head.text == prev_token_text:
            return ""
        return get_noun_dependency(token.head, noun_adj)

    if token.head.text == prev_token_text:
        return ""
    return get_noun_dependency(token.head, noun_adj)


def get_na_pairs(caption):
    doc = nlp(caption.lower())

    noun_adj = {}
    adj = []
    for tok in doc:
        if tok.pos_ == "NOUN":
            noun_adj[tok.text] = []
            if tok.head.pos_ == "VERB":
                noun_adj[tok.text].append(tok.head.text)
        if tok.pos_ == "ADJ":
            adj.append(tok.text)

    for tok in doc:
        # direct dependency
        if tok.pos_ == "ADJ" or tok.pos_ == "PROPN" or tok.pos_ == "VERB":
            noun = get_noun_dependency(tok, noun_adj)
            if noun in noun_adj.keys():
                noun_adj[noun].append(tok.text)

    return noun_adj


# with open("clsWtype.json") as jsonFile:
#     clsWtype = json.load(jsonFile)
#     jsonFile.close()

# for str in clsWtype:
#     print(str)
#     print_pos(str)
#     print()

# print_pos(doc)
# print(get_na_pairs(doc))
