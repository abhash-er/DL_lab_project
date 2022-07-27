from transformers import pipeline
from tkinter import Image

from matplotlib.pyplot import table
from src.defining_attributes import get_att_hierarchy
from na import get_na_pairs
# from transformers import pipeline
import json
from sentence_transformers import SentenceTransformer, util
import numpy as np

# classifier = pipeline("zero-shot-classification")
classifier = pipeline("fill-mask", top_k=300)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
candidate_labels = ["Person", "Vehicle", "Outdoor", "Animal", "Sports", "Kitchenware", "Food", "Furniture", "Electronics",
                    "Appliance", "Indoor", "Cloth"]

attribute_data = get_att_hierarchy()

annotation_list = attribute_data["clsWtype"]

fill_att_dict = {}

cat_supercat_dict = {}


def prefill_attributes():
    global fill_att_dict
    for ann in annotation_list:
        attribute_type, attribute = ann.split(":")
        if attribute_type in fill_att_dict:
            fill_att_dict[attribute_type].update({attribute: -1})
        else:
            fill_att_dict[attribute_type] = {attribute: -1}

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
    for category in instances_val_json["categories"]:
        cat_supercat_dict[category["name"]] = category["supercategory"]
        cat_supercat_dict[category["supercategory"]
                          ] = category["supercategory"]
    # print(cat_supercat_dict)


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
    print()
    sorted_object_list = [x[0] for x in sorted_object_list]
    if sorted_object_list[0] == noun:
        sorted_object_list = sorted_object_list[1:50]
    else:
        sorted_object_list = sorted_object_list[:50]
    return sorted_object_list
    # get max score index


"""
returns list of category_name and supercategory sorted with most similarity
Eg:
caption = A girl is in a kitchen making pizzas
noun = girl
intermediate = A girl is in a kitchen making pizzas
return [{person, person}]
"""


def get_noun_associated_category_name_supercategory(noun, caption):
    masked_caption = caption.replace(noun, "<mask>", 1)
    print()
    print("Noun => "+noun)
    print(masked_caption)
    prepend = None
    if noun in cat_supercat_dict and cat_supercat_dict[noun] == noun:
        prepend = noun
        print("entering")
    object_list = []
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

        print("Possible Choices => ")
        print(possible_words_list)

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
        return [relevant_objects[0], cat_supercat_dict[relevant_objects[0]]]
    else:
        return [None, None]


if __name__ == "__main__":

    with open("captions_val2017.json") as jsonFile:
        captions_val_json = json.load(jsonFile)
        jsonFile.close()

    with open("instances_val2017.json") as jsonFile:
        instances_val_json = json.load(jsonFile)
        jsonFile.close()

    build_cat_supercat_dict()

    #  First get the image from captions file
    index = 0
    img = captions_val_json["images"][index]
    img_captions_list = get_captions(img)
    img_captions_join = '. '.join(img_captions_list)
    # bb_list = get_bbox(img)
    # show_image(img, bb_list)
    print(img_captions_list)
    print(get_na_pairs(img_captions_join))
    na_pair = get_na_pairs(img_captions_join)

    for caption in img_captions_list:
        print()
        print(caption)
        print("="*len(caption))
        possible_instances_bbx = []
        for noun in na_pair:
            if noun in caption:
                cat_name, super_cat = get_noun_associated_category_name_supercategory(
                    noun, caption)
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
