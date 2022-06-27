This README describes the dataset and associated benchmark. By the date of 21.06.2022 this dataset cannot be shared.


=======
Dataset
=======

The dataset is compose of a single file coatt80_val1200.json which contains a dictionary with the following data:

	o info
	General information of the data subject to changes
	
	o images
	List of images on the dataset.
    [
        {
            "id": 15254, % id of the image corresponding to coco 2017 dataset
            "width": 640, % dimensions for the bounding box annotations
            "height": 427, 
            "file_name": "datasets/coco/val2017/000000015254.jpg", % path to the image
            "set": "val300", % set for validation or training the attributes 
        }
        ...
    ]
	
    o categories
    List of 80 object categories 
    [
        {
            'id': 0, 
            'name': 'person',
        }
        ...
    ] 

	o attributes
    List of attributes 121
    [
        {
            "id": 0, 
            "name": "cleanliness:clean/neat/tidy", 
            "type": "cleanliness", 
            "parent_type": "cleanliness", 
            "freq_set": "medium", 
            "is_has_att": "is"
        }
        ...
    ]

    o licenses

    o annotations
    List of object and attribute annotations 
    [
        {
            'id': 1, % id, index of the object, 1 to 8609
            'image_id': 15254, % image of the object, corresponding to MSCOCO ids  
            'bbox': [524.48, 27.41, 98.71, 371.22], % bounding box coordinates [XYWH] 
            'area': 36643.1328125, % area in pixels 
            'iscrowd': 0, 
            'category_id': 44, % object id
            'att_vec': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 0, 0, 1, 0] 
            % attribute annotation: list of 121 integers
            % 1 = positive attribute
            % 0 = negative attribute
            % -1 = ignore attribute
        }
        ...
    ]


==========
Evaluation
==========

For the evaluation we consider two main files: 
- defining_attributes.py
    defines the categories of the attributes, the hierarchy, the synonyms, the types, and ids
    Use the function get_att_hierarchy() to get the following dictionary:

        o hierarchy: attribute hierarchy depending on the object 
        o simple_att: list of attributes with out typeof attribute
        o clsWtype: compleate list of 121 attributes with the type -> this list corresponds to the attribute annotations
        o cls2Id: attribute to id dictionary
        o cls2type: simple_att to type of attribute
        o att_types: list of attribute type
        o is_has_att: split of attribute usage, either "is [attribute]" of "has [attribute]"
        o word2attId: single word attribute 
        o extended_materials: extended list of material categories

- attribute_evaluator.py
    evaluation code to compare performance of methods.
    follow the main function example to evaluate. 


