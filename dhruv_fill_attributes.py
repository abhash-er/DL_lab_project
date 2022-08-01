from transformers import pipeline
from tkinter import Image

from matplotlib.pyplot import table
from src.defining_attributes import get_att_hierarchy
from na import get_na_pairs
# from transformers import pipeline
import json
from sentence_transformers import SentenceTransformer, util
import numpy as np
import gensim.downloader
import sys

# classifier = pipeline("zero-shot-classification")
classifier = pipeline("fill-mask", top_k=300, model="distilroberta-base")
glove_vectors = gensim.downloader.load('glove-wiki-gigaword-50')
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
candidate_labels = ["Person", "Vehicle", "Outdoor", "Animal", "Sports", "Kitchenware", "Food", "Furniture", "Electronics",
                    "Appliance", "Indoor", "Cloth"]

attribute_data = get_att_hierarchy()

annotation_list = attribute_data["clsWtype"]

fill_att_dict = {}

cat_supercat_dict = {}

dict_attr = {}


def prefill_attributes():
    global fill_att_dict
    for ann in annotation_list:
        attribute_type, attribute = ann.split(":")
        if attribute_type in fill_att_dict:
            fill_att_dict[attribute_type].update({attribute: -1})
        else:
            fill_att_dict[attribute_type] = {attribute: -1}


def create_dict_attr():
    global dict_attr
    for ann in annotation_list:
        attribute_type, attribute = ann.split(":")
        if attribute_type not in dict_attr.keys():
            dict_attr[attribute_type] = []
        attributes_list = attribute.split("/")
        for attr in attributes_list:
            dict_attr[attribute_type].append(attr)
        # if len(attributes_list) == 1:
        #     dict_attr[attribute_type].append(attributes_list[0])
        # else:
        #     dict_attr[attribute_type].append(attributes_list)
        # dict_attr[attribute_type].append(attributes_list)
        # att_value_dict = {}
        # for a_value in attribute_values:
        #     att_value_dict[a_value] = -1
        # dict_attr[attribute_type] = att_value_dict
        # if attribute_type in fill_att_dict:
        #     fill_att_dict[attribute_type].update({attribute: -1})
        # else:
        #     fill_att_dict[attribute_type] = {attribute: -1}

        # take category and return associated nouns

        # def get_nouns_cat_list(na_pairs, category):
        #     nouns_cat_list = []
        #     if category == "other":
        #         category = ["Vehicle", "Outdoor", "Sports", "Kitchenware", "Furniture", "Electronics",
        #                     "Appliance", "Indoor"]
        #     for noun in na_pairs:
        #         results = classifier(noun, candidate_labels)
        #         print(results["labels"][0])
        #         if results["labels"][0] in category:
        #             nouns_cat_list.append(noun)
        #         else:
        #             print("{} Not in {}".format(noun, category))
        #     return nouns_cat_list


def get_captions(img):

    # Get captions of the image
    captions_list = []
    ann_list = captions_val_json["annotations"]
    for caption in ann_list:
        if caption["image_id"] == img["id"]:
            captions_list.append(caption["caption"])
    #         pos_dict = print_pos(caption["caption"])
    #         print()
    # pos_dict["NOUN"] = list(set(pos_dict["NOUN"]))
    # pos_dict["LEMMA"] = list(set(pos_dict["LEMMA"]))
    # print(json.dumps(pos_dict, indent=4))
    return captions_list


def build_cat_supercat_dict():
    for category in all_categories_json:
        cat_supercat_dict[category["name"]] = category["supercategory"]
        cat_supercat_dict[category["supercategory"]
                          ] = category["supercategory"]
    # print(cat_supercat_dict)


def get_noun_associated_category_name_supercategory_w2v(noun):
    relevant_objects = []
    results = glove_vectors.most_similar(noun, topn=50)
    # print(len(results))
    print("W2V Possible Choices => ")
    print([x[0] for x in results])
    for key, value in results:
        if key in cat_supercat_dict:
            relevant_objects.append(key)
    print("W2V Relevant Choices => ")
    print(relevant_objects)


def generate_synonmys_for_noun(noun, object_list):
    # Generating similaritiesß
    # print(object_list)
    noun_embedding = model.encode(noun)
    obj_list_embedding = model.encode(object_list)
    # print(util.dot_score(
    #     noun_embedding, obj_list_embedding).tolist()[0])
    similarities = util.dot_score(
        noun_embedding, obj_list_embedding).tolist()[0]
    # Mapping object with its similarity score and sorting
    sorted_object_list = []
    for index, obj in enumerate(object_list):
        sorted_object_list.append((obj, similarities[index]))
    sorted_object_list = sorted(
        sorted_object_list, key=lambda x: x[1], reverse=True)
    sorted_object_list = [x[0] for x in sorted_object_list]
    # if sorted_object_list[0] == noun:
    #     sorted_object_list = sorted_object_list[1:50]
    # else:
    sorted_object_list = sorted_object_list[:50]
    return sorted_object_list
    # get max score index


def all_instances_relatedto_caption(category, supercategory):
    instances_from_caption = []
    for m_category in img_instances_categories_list:
        if m_category["name"] == category and m_category["supercategory"] == supercategory:
            instances_from_caption.append(m_category)
    return instances_from_caption


"""
returns list of category_name and supercategory sorted with most similarity
Eg:
caption = A girl is in a kitchen making pizzas
noun = girl
intermediate = A girl is in a kitchen making pizzas
return [{person, person}]
"""

"""
kitchen
['small', 'has', 'clean', 'ready', 'see', 'has']

Noun/Adj => kitchen
A small <mask> has various appliances and a table.

Possible Choices =>
['kitchen', 'restaurant', 'appliance', 'cafe', 'café', 'oven', 'diner', 'stove', 'house', 'House', 'house', 'cabinet', 'cooker', 'cafeteria', 'bedroom', 'grocery', 'furniture', 'household', 'room', 'room', 'bathroom', 'fridge', 'bowl', 'interior', 'sink',
    'dish', 'place', 'grill', 'cottage', 'refrigerator', 'floor', 'garage', 'supermarket', 'tavern', 'shelf', 'bakery', 'establishment', 'garden', 'chapel', 'home', 'desk', 'studio', 'school', 'microwave', 'wardrobe', 'hut', 'mansion', 'pot', 'business']
['kitchen', 'appliance', 'oven', 'furniture',
    'bowl', 'sink', 'refrigerator', 'microwave']
Closest choice to the noun => kitchen
Category Name => kitchen
Super Category => kitchen
[]

Noun/Adj => small
A <mask> kitchen has various appliances and a table.
prepend => small

Possible Choices =>
['small', 'tiny', 'smaller', 'large', 'larger', 'little', 'miniature', 'bigger', 'big', 'huge', 'mini', 'massive', 'giant', 'vast', 'spacious', 'round', 'suitable', 'compact', 'whole', 'regular', 'pocket', 'fitted', 'usable', 'modified',
    'simple', 'standard', 'spare', 'black', 'bare', 'white', 'colorful', 'square', 'perfect', 'fancy', 'toy', 'cramped', 'minimalist', 'comparable', 'mixed', 'used', 'clean', 'heated', 'basic', 'style', 'main', 'nearby', 'full', 'decent', 'modest']
['small', 'tiny', 'little']
Closest choice to the noun => small
Category Name => small
size
little
Adjective => small
Closest Value :
Adj type => size
Adj value => little

"""


def get_noun_associated_category_name_supercategory(noun, caption):
    masked_caption = caption.replace(noun, "<mask>", 1)
    print("-----------------------------------------------------------------")
    print("Noun => "+noun)
    print(masked_caption)
    prepend = None
    if noun in cat_supercat_dict and cat_supercat_dict[noun] == noun:
        prepend = noun

    if "<mask>" in masked_caption:
        results = classifier(masked_caption)
        possible_words_list = []
        if prepend is not None:
            possible_words_list.append(prepend)
        for words in results:
            possible_words = words["token_str"].strip()
            possible_words_list.append(possible_words)
            # print(possible_words)
        possible_words_list = generate_synonmys_for_noun(
            noun, possible_words_list)
        # if possible_words in cat_supercat_dict:
        #     object_list.append(possible_words)

        # print("Possible Choices => ")
        # print(possible_words_list)

        relevant_objects = []
        for possible_word in possible_words_list:
            if possible_word in cat_supercat_dict:
                relevant_objects.append(possible_word)
        print(relevant_objects)

        if len(relevant_objects) == 0:
            return [None, None]

    #     # testing similarity
    #     # noun = "girl"
    #     # object_list = ["pizza", "person", "woman", "food"]
    #     # print(object_list)

        # Generating similaritiesß
        # noun_embedding = model.encode(noun)
        # obj_list_embedding = model.encode(object_list)
        # print(util.dot_score(
        #     noun_embedding, obj_list_embedding).tolist()[0])
        # similarities = util.dot_score(
        #     noun_embedding, obj_list_embedding).tolist()[0]

    #     # Mapping object with its similarity score and sorting
    #     # sorted_object_list = []
    #     # for index, obj in enumerate(object_list):
    #     #     sorted_object_list.append((obj, similarities[index]))
    #     # sorted_object_list = sorted(
    #     #     sorted_object_list, key=lambda x: x[1], reverse=True)
    #     # print()
    #     # print(sorted_object_list)
    #     # get max score index
    #     index_max = np.argmax(similarities)
    # #     # print(index_max)
    #     relevant_object = object_list[index_max]
        print("Closest choice to the noun => "+relevant_objects[0])
        print("Category Name => "+relevant_objects[0])

        print("Super Category => "+cat_supercat_dict[relevant_objects[0]])
        # Printing all instances relating to the corresponding caption
        print(all_instances_relatedto_caption(
            relevant_objects[0], cat_supercat_dict[relevant_objects[0]]))
        return [relevant_objects[0], cat_supercat_dict[relevant_objects[0]]]
    else:
        return [None, None]


def get_bbx_img_id(img):
    instance_list = []
    for instance in instances_val_json["annotations"]:
        if instance["image_id"] == img["id"]:
            # print(instance)
            instance_list.append(instance)

    return instance_list


def get_img_instances_categeories(instances_list):
    instances_category_list = []
    for instance in instances_list:
        instances_category_list.append(
            all_categories_array[instance["category_id"]])

    return instances_category_list


def make_all_categories_array(all_categories_json):
    categories_list = ["" for i in range(all_categories_json[-1]["id"]+1)]
    for category in all_categories_json:
        categories_list[category["id"]] = category

    return categories_list


def search_in_attr_dict(word):
    att_list = []
    for attr_type in dict_attr:
        # print(attr_type)
        # print(attr_values)
        # for attr in attr_values.split("/"):
        
        if word.lower() in dict_attr[attr_type]:
            att_list.append({attr_type, word.lower()})
    if len(att_list) != 0:
        print(att_list)
    return att_list


def get_adj_associated_attribute_and_type(adj, caption):
    print("- "*20)
    masked_caption = caption.replace(adj, "<mask>", 1)
    print()
    print("Adjective => "+adj)

    if "<mask>" in masked_caption:
        print(masked_caption)
        results = classifier(masked_caption)
        possible_words_list = []
        for words in results:
            possible_words = words["token_str"].strip()
            possible_words_list.append(possible_words)

        possible_words_list = generate_synonmys_for_noun(
            adj, possible_words_list)

        possible_words_list = possible_words_list[:10]
        print("Possible Choices => ")
        print(possible_words_list)

        relevant_attr = []
        for possible_word in possible_words_list:
            att_list = search_in_attr_dict(possible_word)
            if len(att_list) != 0:
                # print(att_list)
                relevant_attr.append(att_list)

        # print(relevant_attr)

        if len(relevant_attr) == 0:
            return [None, None]

        #     if len(relevant_objects) == 0:
        #         return [None, None]

        # #     # testing similarity
        # #     # noun = "girl"
        # #     # object_list = ["pizza", "person", "woman", "food"]
        # #     # print(object_list)

        #     # Generating similaritiesß
        #     # noun_embedding = model.encode(noun)
        #     # obj_list_embedding = model.encode(object_list)
        #     # print(util.dot_score(
        #     #     noun_embedding, obj_list_embedding).tolist()[0])
        #     # similarities = util.dot_score(
        #     #     noun_embedding, obj_list_embedding).tolist()[0]

        # #     # Mapping object with its similarity score and sorting
        # #     # sorted_object_list = []
        # #     # for index, obj in enumerate(object_list):
        # #     #     sorted_object_list.append((obj, similarities[index]))
        # #     # sorted_object_list = sorted(
        # #     #     sorted_object_list, key=lambda x: x[1], reverse=True)
        # #     # print()
        # #     # print(sorted_object_list)
        # #     # get max score index
        # #     index_max = np.argmax(similarities)
        # # #     # print(index_max)
        # #     relevant_object = object_list[index_max]
        print(relevant_attr)
        # print("Closest choice to the Adj => "+relevant_attr[0][1])
        # print("Attribute Value => "+relevant_attr[0][1])

        # print("Attribute Type => "+relevant_attr[0][0])
        # # Printing all instances relating to the corresponding caption
        # # print(all_instances_relatedto_caption(
        # #     relevant_objects[0], cat_supercat_dict[relevant_objects[0]]))
        # return [relevant_attr[0][0], relevant_attr[0][1]]
    else:
        print("Skipping adjective as not present in the sentence...")
        return [None, None]
    return [None, None]


if __name__ == "__main__":

    with open("captions_val2017.json") as jsonFile:
        captions_val_json = json.load(jsonFile)
        jsonFile.close()

    with open("instances_val2017.json") as jsonFile:
        instances_val_json = json.load(jsonFile)
        jsonFile.close()

    prefill_attributes()
    # print(fill_att_dict)

    create_dict_attr()
    print(json.dumps(dict_attr, indent=4))

    all_categories_json = instances_val_json["categories"]
    all_categories_array = make_all_categories_array(all_categories_json)

    build_cat_supercat_dict()
    # print(json.dumps(cat_supercat_dict, indent=4, sort_keys=True))

    # get_adj_associated_attribute_and_type(
    #     "small", "A small kitchen has various appliances and a table")

    #  First get the image from captions file
    index = 0
    for index in range(10):
        # if index == 2:
        #     break
        img = captions_val_json["images"][index]
        img_captions_list = get_captions(img)
        img_captions_join = '. '.join(img_captions_list)
        # bb_list = get_bbox(img)
        # show_image(img, bb_list)
        print(img_captions_list)
        print(get_na_pairs(img_captions_join))
        na_pair = get_na_pairs(img_captions_join)

        instances_bbx = get_bbx_img_id(img)
        img_instances_categories_list = get_img_instances_categeories(
            instances_bbx)
        print(img_instances_categories_list)

        for caption in img_captions_list:
            print()
            print(caption)
            print("="*len(caption))

            for noun, adjs in na_pair.items():
                if noun in caption and len(na_pair[noun]) != 0:
                    cat_name, super_cat = get_noun_associated_category_name_supercategory(
                        noun, caption)
                    if cat_name is not None and super_cat is not None:
                        for adj in adjs:
                            att_type, att_value = get_adj_associated_attribute_and_type(
                                adj, caption)

    #     print("Adjective => "+str(adj))
    #     print("Closest Value : ")
    #     print("Adj type => "+str(att_type))
    #     print("Adj value => "+str(att_value))
    #     print("-----------------------------------------")

    # get_noun_associated_category_name_supercategory_w2v(noun)
    # if cat_name is not None and super_cat is not None:
    #     possible_instances_bbx.append({super_cat: cat_name})
    # cat_name, super_cat = get_noun_associated_category_name_supercategory(
    #     "pans", "Man in apron standing on front of oven with pans and bakeware")
    # print(cat_name)
    # print(super_cat)

    # classifier = pipeline("fill-mask", top_k=10)
    # resulta = classifier(
    #     "A <mask> is in a kitchen making pizzas")
    # print(resulta)
