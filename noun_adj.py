import json
from PIL import Image
import matplotlib.pyplot as plt
import spacy
import pandas as pd
import matplotlib.patches as patches


def print_pos(str):

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(str)
    co_dict = {
        "token.text": [],
        "token.lemma_": [],
        "token.pos_": [],
        "token.tag_": [],
        "token.dep_": [],
        "token.shape_": [],
        "token.is_alpha": [],
        "token.is_stop": []
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

        if token.pos_ == "NOUN":
            pos_dict["NOUN"].append(token.text)
            pos_dict["LEMMA"].append(token.lemma_)

    df = pd.DataFrame(data=co_dict)
    print(df)
    # print(json.dumps(pos_dict, indent=4))

    return pos_dict

    # for token in doc:
    #     if co_dict["token.pos"] == "NOUN":


def get_captions_pos(img):
    # global pos_dict

    # Get captions of the image
    ann_list = captions_val_json["annotations"]
    for caption in ann_list:
        if caption["image_id"] == img["id"]:
            print(caption)
            print()
            pos_dict = print_pos(caption["caption"])
            print()
    pos_dict["NOUN"] = list(set(pos_dict["NOUN"]))
    pos_dict["LEMMA"] = list(set(pos_dict["LEMMA"]))
    print(json.dumps(pos_dict, indent=4))
    return pos_dict


def get_bbox(img):

    # Getting all bbox of an image
    img_ann_list = []
    for ann_obj in instances_val_json["annotations"]:
        if ann_obj["image_id"] == img["id"]:
            img_ann_list.append(ann_obj)
    return img_ann_list


def show_image(img, img_ann_list):

    for ann in img_ann_list:
        bbox = ann["bbox"]
        for category in instances_val_json["categories"]:
            if category["id"] == ann["category_id"]:
                print(category)
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1,
                                 edgecolor='r', facecolor='none')

        im = Image.open("src/datasets/coco/val2017/"+img['file_name'])

        # Create figure and axes
        fig, ax = plt.subplots()

        # Display the image
        ax.imshow(im)
        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.show()


if __name__ == "__main__":

    captions_val_json = {}
    instances_val_json = {}
    pos_dict = {"NOUN": [], "LEMMA": []}
    # load all json
    with open("captions_val2017.json") as jsonFile:
        captions_val_json = json.load(jsonFile)
        jsonFile.close()
    with open("instances_val2017.json") as jsonFile:
        instances_val_json = json.load(jsonFile)
        jsonFile.close()

    #  First get the image from captions file
    index = 0
    img = captions_val_json["images"][index]
    img_captions_pos = get_captions_pos(img)
    bb_list = get_bbox(img)
    show_image(img, bb_list)
