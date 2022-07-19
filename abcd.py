import matplotlib.patches as patches
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from unicodedata import category
from PIL import Image
import json
from src.defining_attributes import get_att_hierarchy


# with open("captions_val2017.json") as jsonFile:
#     data = json.load(jsonFile)
#     jsonFile.close()

# for ob in data["annotations"]:
#     if ob["image_id"] == 179765:
#         print(ob)


with open("coatt80_val1200.json") as jsonFile:
    data = json.load(jsonFile)
    jsonFile.close()


def show_img_stuff(index):

    # Get image from annotations
    annotation_0 = data["annotations"][index]
    img_id = annotation_0["image_id"]
    image = {}
    for img_obj in data["images"]:
        if img_obj["id"] == img_id:
            image = img_obj
            print(img_obj)  # return 1
    bbox = annotation_0["bbox"]
    category_id = annotation_0["category_id"]
    for cat_id in data["categories"]:
        if cat_id["id"] == category_id:
            print(cat_id)  # return 1
            category = cat_id

    att_vec = annotation_0["att_vec"]
    print(att_vec)
    classWtype = get_att_hierarchy()["clsWtype"]
    cls2id = get_att_hierarchy()["cls2Id"]
    print("clsWtype")
    print("===========")
    for i in range(len(att_vec)):
        if att_vec[i] == 1:
            print(str(att_vec[i])+"\t"+classWtype[i] +
                  "\t"+str(cls2id[classWtype[i]]))

    im = Image.open("src/"+image['file_name'])

    # Create figure and axes
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(im)

    # Create a Rectangle patch
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1,
                             edgecolor='r', facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)

    plt.show()


for i in range(8, 15):
    show_img_stuff(i)
