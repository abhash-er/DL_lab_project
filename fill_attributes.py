from na import get_na_pairs
from src.defining_attributes import get_att_hierarchy
import spacy
from transformers import pipeline

classifier = pipeline("zero-shot-classification")
candidate_labels = ["person", "animal", "food", "other", "cloth"]

nlp = spacy.load('en_core_web_sm')
attribute_data = get_att_hierarchy()
att_hierarchy = attribute_data["hierarchy"]
human_categories = att_hierarchy["human"]
food_categories = att_hierarchy["food"]
animal_categories = att_hierarchy["animal"]
general_categories = att_hierarchy["object"]
extended_materials = attribute_data["extended_materials"]

# change the animal materials and general_category materials
new_material_list = []
for material_type in general_categories["material"]:
    material_string = "/".join(extended_materials[material_type])
    new_material_list.append(material_string)

general_categories["material"] = new_material_list
animal_categories["material"] = new_material_list

def isPresent(search_list, parent_list):
    for search_key in search_list:
        if search_key in parent_list:
            return True
    return False


def get_nouns_cat_list(na_pairs, category):
    nouns_cat_list = []
    for noun in na_pairs:
        results = classifier(noun, candidate_labels)
        if results["labels"][0] == category:
            nouns_cat_list.append(noun)
        else: 
            print("{} Not in {}".format(noun, category))
    return nouns_cat_list


def fill_opposite_attributes(fill_att_dict, category):
    categories = None
    if category == "person":
        categories = human_categories
    elif category == "animal":
        categories = animal_categories
    elif category == "food":
        categories = food_categories
    else:
        categories = general_categories

    for attribute_type in categories:
        # first check whether a flag is up in the dictionary
        is_flag_up = False
        for att in categories[attribute_type]:
            if fill_att_dict[attribute_type][att] == 1:
                is_flag_up = True

        if is_flag_up:
            for att in categories[attribute_type]:
                if fill_att_dict[attribute_type][att] == -1:
                    fill_att_dict[attribute_type][att] = 0

    return fill_att_dict


att2type = attribute_data["cls2type"]
annotation_list = attribute_data["clsWtype"]
attributes = attribute_data["simple_att"]

att_list = []

fill_att_dict = {}
for ann in annotation_list:
    attribute_type, attribute = ann.split(":")
    if attribute_type in fill_att_dict:
        fill_att_dict[attribute_type].update({attribute: -1})
    else:
        fill_att_dict[attribute_type] = {attribute: -1}

caption = "A happy man with his sad wife"
na_pairs = get_na_pairs(caption)
print(na_pairs)
# get object from bbox
object_category = "person"

# Objective: Get the object from bbox and check which of the attribute matches with it
if object_category == "person":
    nouns_to_check = get_nouns_cat_list(na_pairs, "person")
    cloth_nouns = get_nouns_cat_list(na_pairs, "cloth")

    for attribute_type in human_categories:
        for att in human_categories[attribute_type]:
            # Check for hair color
            search_keys = att.split("/")
            if "hair" in attribute_type:
                for noun in nouns_to_check:
                    if "haired" in na_pairs[noun] and isPresent(search_keys, na_pairs[noun]):
                        fill_att_dict[attribute_type][att] = 1

            # Check for cloth color
            elif "cloth" in attribute_type:
                for noun in cloth_nouns:
                    if isPresent(search_keys, na_pairs[noun]):
                        fill_att_dict[attribute_type][att] = 1
            # general
            else:
                for noun in nouns_to_check:
                    if isPresent(search_keys, na_pairs[noun]):
                        fill_att_dict[attribute_type][att] = 1
                # also check in the nouns
                if isPresent(search_keys, na_pairs.keys()):
                    fill_att_dict[attribute_type][att] = 1
    fill_att_dict = fill_opposite_attributes(fill_att_dict, "person")

else:
    # rest of categories are straight forward
    nouns_to_check = get_nouns_cat_list(na_pairs, object_category)
    categories = None
    if object_category == "animal":
        categories = animal_categories
    elif object_category == "food":
        categories = food_categories
    else:
        categories = general_categories

    for attribute_type in categories:
        for att in categories[attribute_type]:
            search_keys = att.split("/")
            for noun in nouns_to_check:
                if isPresent(search_keys, na_pairs[noun]):
                    fill_att_dict[attribute_type][att] = 1

            if isPresent(search_keys, na_pairs.keys()):
                fill_att_dict[attribute_type][att] = 1

    fill_att_dict = fill_opposite_attributes(fill_att_dict, object_category)

for key in fill_att_dict:
    print(key)
    print(fill_att_dict[key])
    print()


# TODO after the attribute list is filled, check for the color_quantity by keeping a count for color
# TODO from bbox check isCrowd attribute to fill the group attribute in the attribute list
# TODO make vector out of this
# TODO Integrate with bbox and captions (make this file function)

# TODO make a model skeleton for the ICM loss