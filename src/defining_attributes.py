# References
# https://www.collinsdictionary.com/
# https://www.oed.com/
# https://www.wikidata.org/wiki/Wikidata:Main_Page
# soy visual https://www.soyvisual.org/
#
# Dictionary of attributes
"""
https://www.wikidata.org
attribute: quality, character, or characteristic ascribed to someone or something; 
object closely associated with or belonging to a specific person, thing, or office

property: predominant feature that characterizes a being, a thing, a phenomenon, 
etc. and which differentiates one being from another, one thing from another
"""

# Root level
general_categories = {
    "color": [
        "black",
        "white",
        "gray",
        "tan",
        "brown",
        "green",
        "red",
        "yellow",
        "blue",
        "orange",
        "violet",
        "pink",
    ],
    "color quantity": ["single-colored/unicolored", "two-colored", "multicolored/colorful"],
    "tone": ["light/bright/shiny", "dark/obscure"],
    "optical property": [
        "transparent/translucent",
        "opaque",
        "reflective",
    ],
    "pattern": [
        "plain",
        "striped/lined/pinstriped",
        "plaid/tartan",
        "dotted/speckled/spotted",
        "floral",
        "checkered",
        "tiled",
        "lettered",
    ],
    "size": ["big/large/giant/huge", "small/little/tiny"],
    "texture": ["smooth/sleek", "soft/fluffy/flossy/furry/hairy", "tough/hard/rough"],
    "material": [
        "metallic",
        "wood",
        "ceramic",
        "paper",
        "textile",
        "glass",
        "leather",
        "stone",
        "polymers",
        "construction material",
    ],
    "position": ["horizontal/lying", "vertical/upright/standing"],
    "group": ["single", "group"],
    "cleanliness": ["clean/neat/tidy", "unclean/untidy/dirt/dirty/muddy"],
    "order": [
        "messy/disordered/unordered/disorganized/unorganized/cluttered",
        "ordered/arranged/organized",
    ],
    "length": ["long", "short"],
    "state": [
        "empty",
        "full/whole",
        "piece/cut",
        "closed",
        "open",
        "cracked",
        "covered",
        "dilapidated/ruined/broken",
        "dry",
        "wet",
        "folded",
        "on",
        "off",
    ]
}

extended_materials = {
    "metallic": [
        "metal/metallic",
        "aluminum",
        "brass/copper-zinc",
        "iron",
        "stainless steel",
        "steel",
        "silver",
    ],
    "wood": ["wood/wooden", "bamboo", "hardwood"],
    "ceramic": ["ceramic", "brick", "ceramic", "porcelain"],
    "paper": ["paper", "cardboard"],
    "textile": [
        "textile",
        "cloth",
        "fabric",
        "denim",
        "cotton",
        "jean",
        "silk",
        "plush",
    ],
    "glass": ["glass"],
    "leather": ["leather"],
    "stone": [
        "stone",
        "granite",
        "cobblestone",
        "gravel",
        "marble",
        "pebbled",
        "rocky",
        "sandy",
    ],
    "polymers": ["polymers", "plastic", "rubber", "styrofoam", "polymer"],
    "construction material": ["asphalt", "cement", "clay", "concrete", "stucco"],
}

food_categories = {
    "color": general_categories["color"],
    "color quantity": general_categories["color quantity"],
    "tone": general_categories["tone"],
    "size": general_categories["size"],
    "texture": general_categories["texture"],
    "group": general_categories["group"],
    "order": general_categories["order"],
    "state": ["full/whole", "piece/cut"],
    "cooked": ["cooked/baked/warmed", "raw/fresh"],
}

animal_categories = {
    "color": general_categories["color"],
    "color quantity": general_categories["color quantity"],
    "tone": general_categories["tone"],
    "pattern": general_categories["pattern"],
    "size": general_categories["size"],
    "texture": general_categories["texture"],
    "material": general_categories["material"],
    "position": general_categories["position"] + ["sitting/sit"],
    "group": general_categories["group"],
    "cleanliness": general_categories["cleanliness"],
    "maturity": ["young/baby", "adult/old/aged"],
    "state": ["dry", "wet"],
}

human_categories = {
    # ["baby/kid/child/boy/girl", "teen/adult/elder"],
    "maturity": animal_categories["maturity"],
    "gender": ["male/man/guy/boy", "female/woman/girl"],
    "hair length": general_categories["length"] + ["bald"],
    "hair color": general_categories["color"],
    "hair tone": general_categories["tone"],
    "hair type": ["straight", "curly/curled"],
    "skin tone": general_categories["tone"],
    "face expression": [
        "angry/mad",
        "disgust/frowning",
        "neutral/calm/serious",
        "fear/surprise",
        "happy/smiling/laughing/grinning/joyful",
        "sad/unhappy",
        "sleepy/sleeping",
    ],
    "clothes color": general_categories["color"],
    "clothes pattern": general_categories["pattern"],
    "position": animal_categories["position"],
    "group": general_categories["group"],
}

is_has_att = {
    "is": [
        "color",
        "color quantity",
        "tone",
        "optical property",
        "pattern",
        "size",
        "width",
        "depth",
        "texture",
        "material",
        "position",
        "group",
        "cleanliness",
        "order",
        "newness",
        "length",
        "state",
        "maturity",
        "gender",
        "height",
        "thickness",
        "cooked",
    ],
    "has": [
        "hair length",
        "hair color",
        "hair tone",
        "hair type",
        "skin tone",
        "face expression",
        "clothes color",
        "clothes pattern",
    ],
}


def get_att_hierarchy():
    attribute_hierarchy = {
        "object": general_categories,
        "human": human_categories,
        "animal": animal_categories,
        "food": food_categories,
    }

    # build att2type dict
    attWtype = []
    attributes = []
    att_types = []
    att2type = {}
    for obj_type, att_categories in attribute_hierarchy.items():
        for type_a, list_a in att_categories.items():
            if type_a == "material":
                list_a = ["/".join(extended_materials[material])
                          for material in list_a]
            attributes += list_a
            attWtype += [type_a + ":" + att for att in list_a]
            for att in list_a:
                if att not in att2type.keys():
                    att2type[att] = []
                att2type[att].append(type_a)
            att_types.append(type_a)

    att_types = list(set(att_types))
    att_types.sort()
    attributes = list(set(attributes))
    attributes.sort()
    attWtype = list(set(attWtype))
    attWtype.sort()
    att_cls2Id = {att: idx for idx, att in enumerate(attWtype)}

    word2attId = {}
    for list_a in attWtype:
        for w_att in list_a.split(":")[-1].split("/"):
            if w_att not in word2attId.keys():
                word2attId[w_att] = []
            # print("w_att: ", w_att)
            # print("word2attId so far: ",word2attId)
            # print(list_a)
            # print("yk:",att_cls2Id[list_a])
            # print(attWtype)
            word2attId[w_att].append(att_cls2Id[list_a])

    attribute_data = {
        "hierarchy": attribute_hierarchy,
        "simple_att": attributes,
        "clsWtype": attWtype,
        "cls2Id": att_cls2Id,
        "cls2type": att2type,
        "att_types": att_types,
        "is_has_att": is_has_att,
        "word2attId": word2attId,
        "extended_materials": extended_materials,
    }
    return attribute_data


get_att_hierarchy()
