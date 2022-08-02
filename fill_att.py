import json
import os
import pickle

import cv2
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data.dataset import Dataset
from src.defining_attributes import get_att_hierarchy
from na import get_na_pairs
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ColorJitter, RandomHorizontalFlip
from torch.utils.data import DataLoader

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def check_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def _convert_image_to_rgb(image):
    return image.convert("RGB")


classifier_zs = pipeline("zero-shot-classification")
classifier = pipeline("fill-mask", top_k=300)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

attribute_data = get_att_hierarchy()
att_hierarchy = attribute_data["hierarchy"]
human_categories = att_hierarchy["human"]
food_categories = att_hierarchy["food"]
animal_categories = att_hierarchy["animal"]
general_categories = att_hierarchy["object"]
extended_materials = attribute_data["extended_materials"]

new_material_list = []
for material_type in general_categories["material"]:
    material_string = "/".join(extended_materials[material_type])
    new_material_list.append(material_string)

general_categories["material"] = new_material_list
animal_categories["material"] = new_material_list

annotation_list = attribute_data["clsWtype"]
attributes = attribute_data["simple_att"]

cat_supercat_dict = {}


def isPresent(search_list, parent_list):
    """
   Check if the search key is present in the parent list
   Helper function for occurrence of attributes
   """
    for search_key in search_list:
        if search_key in parent_list:
            return True
    return False


def get_nouns_cat_list(na_pairs, category):
    """
       FUnction to get nouns which are associated with a category
       It is only being used for checking cloth based nouns
   """
    nouns_cat_list = []
    candidate_labels = ["person", "animal", "food", "other", "cloth"]
    for noun in na_pairs:
        results = classifier_zs(noun, candidate_labels)
        if results["labels"][0] == category:
            nouns_cat_list.append(noun)
        else:
            # print("{} Not in {}".format(noun, category))
            pass
    return nouns_cat_list


def fill_opposite_attributes(fill_att_dict, category):
    """
    This function helps marking negative attributes after all positive attributes gets filled
   """
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


def get_nouns(noun_info, bbox_cat):
    """
   Given a noun info and bounding box object name, return all related nouns related to the object name
   """
    rel_nouns = []
    for noun in noun_info:
        if noun_info[noun]["object_cat"] == bbox_cat:
            rel_nouns.append(noun)
    return rel_nouns


def check_attributeFill(att_dict):
    """
   A safe check to see if any attirbute at all is marked or not
   """
    empty = True
    for att_type in att_dict:
        for att in att_dict[att_type]:
            if att_dict[att_type][att] == 1:
                empty = False
    return empty


def mark_multicolor_att(att_dict):
    '''
    Function to mark color quantity attributes
    '''
    color_quantity = 0

    for att in att_dict["color"]:
        if att_dict["color"][att] == 1:
            color_quantity += 1

    if color_quantity == 1:
        att_dict["color quantity"]["single-colored/unicolored"] = 1
    elif color_quantity == 2:
        att_dict["color quantity"]["two-colored"] = 1
    elif color_quantity > 2:
        att_dict["color quantity"]["multicolored/colorful"] = 1

    return att_dict


def get_attribute_dict(bbox_cat, bbox_supercat, na_pairs, noun_info):
    """
   inputs:
       bbox_cat : the object name of the bbox
       bbox_supercat: supercategory of the bbox
       na_pairs: noun-attribute dictionary
       noun_info: [noun: cat, supercat] information dictionary
   return: attribute dictionary
   """
    fill_att_dict = {}
    for ann in annotation_list:
        attribute_type, attribute = ann.split(":")
        if attribute_type in fill_att_dict:
            fill_att_dict[attribute_type].update({attribute: -1})
        else:
            fill_att_dict[attribute_type] = {attribute: -1}

    nouns_to_check = get_nouns(noun_info, bbox_cat)
    if bbox_supercat == "person":
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

        # Reverse opposite attributes
        fill_att_dict = fill_opposite_attributes(fill_att_dict, "person")

    else:
        # for rest of super categories
        categories = None
        if bbox_supercat == "animal":
            categories = animal_categories
        elif bbox_supercat == "food":
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

        fill_att_dict = fill_opposite_attributes(fill_att_dict, bbox_supercat)

    fill_att_dict = mark_multicolor_att(fill_att_dict)

    return fill_att_dict


def get_attribute_vector(attribute_dict):
    """
   Build attribute vector from the attribute dictionary
   """
    attribute_vector = []
    for ann in annotation_list:
        attribute_type, attribute = ann.split(":")
        label = attribute_dict[attribute_type][attribute]
        label = 2 if label == -1 else label
        attribute_vector.append(label)

    return attribute_vector


def get_captions(captions_json, img_id):
    """
   Get all captions associated with the image given image id
   """
    captions_list = []
    ann_list = captions_json["annotations"]
    for caption in ann_list:
        if caption["image_id"] == img_id:
            captions_list.append(caption["caption"])
    return captions_list


def build_cat_supercat_dict(instance_json):
    """
   Build a global category: super category dictionary
   """
    for category in instance_json["categories"]:
        cat_supercat_dict[category["name"]] = category["supercategory"]
        cat_supercat_dict[category["supercategory"]] = category["supercategory"]


def generate_synonmys_for_noun(noun, object_list):
    """
   Function to generate similarity between noun and object list
   """
    # Generating similarities
    # print(object_list)
    noun_embedding = model.encode(noun)
    obj_list_embedding = model.encode(object_list)

    similarities = util.dot_score(noun_embedding, obj_list_embedding).tolist()[0]

    # Mapping object with its similarity score and sorting
    sorted_object_list = []
    for index, obj in enumerate(object_list):
        sorted_object_list.append((obj, similarities[index]))
    sorted_object_list = sorted(
        sorted_object_list, key=lambda x: x[1], reverse=True)
    # print()
    sorted_object_list = [x[0] for x in sorted_object_list]
    if sorted_object_list[0] == noun:
        sorted_object_list = sorted_object_list[1:50]
    else:
        sorted_object_list = sorted_object_list[:50]
    return sorted_object_list
    # get max score index


def get_noun_associated_category_name_supercategory(noun, caption):
    masked_caption = caption.replace(noun, "<mask>", 1)
    # print()
    # print("Noun => " + noun)
    # print(masked_caption)
    prepend = None
    if noun in cat_supercat_dict and cat_supercat_dict[noun] == noun:
        prepend = noun
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

        # print("Possible Choices => ")
        # print(possible_words_list)

        relevant_objects = []
        for possible_word in possible_words_list:
            if possible_word in cat_supercat_dict:
                relevant_objects.append(possible_word)
        # print(relevant_objects)

        if len(relevant_objects) == 0:
            return [None, None]

        # print("Closest choice to the noun => " + relevant_objects[0])
        # print("Category Name => " + relevant_objects[0])
        # print("Super Category => " + cat_supercat_dict[relevant_objects[0]])
        return [relevant_objects[0], cat_supercat_dict[relevant_objects[0]]]
    else:
        return [None, None]


def get_cat_supercat(categories, id):
    """
   Given a category id, return category name and it's supercategory
   """
    for cat in categories:
        if cat["id"] == id:
            return cat["name"], cat["supercategory"]


class CaptionDataset(Dataset):
    def __init__(self, caption_json, instances_json, transform=None):
        self.images_map = caption_json["images"]
        self.caption_json = caption_json
        self.instance_json = instances_json

        if transform is not None:
            self.transform = transform
        else:
            self.transform = Compose([
                Resize((224, 224), interpolation=BICUBIC),
                ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                # CenterCrop(224),
                RandomHorizontalFlip(),
                _convert_image_to_rgb,
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])

        self.images = []
        self.attribute_vectors = []
        self.root = "src/datasets/coco/train2017"
        self.out_path = "src/datasets/cap/train/"
        check_dir(self.out_path)
        self.att_idx = 0
        if not os.path.exists("src/datasets/cap/train_labels.pkl"):
            for idx in range(len(self.images_map)):
                self.fill_items(idx)
                with open("src/datasets/cap/train_labels.pkl", 'wb') as t:
                    pickle.dump(self.attribute_vectors, t)
        else:
            with open("src/datasets/cap/train_labels.pkl", 'rb') as f:
                self.attribute_vectors = pickle.load(f)

    def fill_items(self, idx):
        img_path = os.path.join(self.root, self.images_map[idx]['file_name'])
        image_id = self.images_map[idx]['id']
        img = cv2.imread(img_path)  # complete this

        # get all captions and make na_pairs
        caption_list = get_captions(self.caption_json, image_id)
        captions = '. '.join(caption_list)
        na_pair = get_na_pairs(captions)
        noun_info = {}
        for caption in caption_list:
            for noun in na_pair:
                if noun in caption:
                    cat_name, super_cat = get_noun_associated_category_name_supercategory(noun, caption)
                    noun_info[noun] = {"object_cat": cat_name, "super_cat": super_cat}
        
        # now check for each instance category and super category and fill attributes
        categories = self.instance_json["categories"]
        print("=" * 10)
        # print(na_pair)
        print("image_id ", image_id)
        bbox_attributes = []
        for bb in self.instance_json["annotations"]:
            if bb['area'] < 4000:
                continue
            if bb["image_id"] == image_id:
                is_crowd = bool(bb["iscrowd"])
                bb_cat, bb_supercat = get_cat_supercat(categories, bb["category_id"])
                att_dict = get_attribute_dict(bb_cat, bb_supercat, na_pair, noun_info)
                if is_crowd:
                    att_dict["group"]["group"] = 1
                    att_dict["group"]["single"] = 0
                else:
                    if bb_supercat in ["person", "animal"]:
                        att_dict["group"]["group"] = 0
                        att_dict["group"]["single"] = 1

                is_empty = check_attributeFill(att_dict)
                if is_empty:
                    continue
                att_vector = get_attribute_vector(att_dict)
                bb_image = self.get_image_crop(bb["bbox"], img)
                out_name = self.out_path + "img_" + str(self.att_idx) + '.png'
                cv2.imwrite(out_name, bb_image)
                self.attribute_vectors.append(att_vector)
                bbox_attributes.append(att_vector)
                self.att_idx += 1
                # print("+++++++++++++++++++")
                # print("bb_cat", bb_cat)
                # print("bb_supercat", bb_supercat)
                # print(json.dumps(att_dict, indent=4))
                # print("+++++++++++++++++++")
                # print()

        whole_att = []
        for i in range(121):
            att_flag = -1
            is_zero = False
            for att_vec in bbox_attributes:
                if att_vec[i] == 1:
                    att_flag = 1
                    break
                elif att_flag == 0:
                    is_zero = True 
            
            if att_flag == -1 and is_zero:
                att_flag = 0 
            whole_att.append(att_flag)
        
        out_name = self.out_path + "img_" + str(self.att_idx) + '.png'
        cv2.imwrite(out_name, img)
        self.attribute_vectors.append(whole_att)
        self.att_idx += 1
                    

    def get_image_crop(self, bbox, img):
        x, y, w, h = bbox
        x, y, w, h = int(x), int(y), int(w), int(h)
        left, upper = x, y
        right, lower = x + w, y + h
        return img[y:y + h, x:x + w]
    
    def save_attribute_vec_list(self):
        with open("src/datasets/cap/train_labels.pkl", 'wb') as t:
                pickle.dump(self.attribute_vectors, t)

    def __getitem__(self, idx):
        img_name = 'img_' + str(idx) + '.png'
        img_path = os.path.join(self.out_path, img_name)
        img = Image.open(img_path)
        img = self.transform(img)
        # plt.imshow(img)
        # plt.show()
        # print(self.attribute_vectors[idx])
        # print()
        return img, self.attribute_vectors[idx]

    def __len__(self):
        return len(self.attribute_vectors)


if __name__ == "__main__":
    with open("src/annotations/captions_train2017.json") as jsonFile:
        caption_json = json.load(jsonFile)
        jsonFile.close()

    with open("src/annotations/instances_train2017.json") as jsonFile:
        instances_json = json.load(jsonFile)
        jsonFile.close()

    build_cat_supercat_dict(instances_json)
    
    training_dataset = CaptionDataset(caption_json, instances_json)
    train_dataloader = DataLoader(training_dataset, batch_size=1, shuffle=True)

    for image, label in train_dataloader:
        continue