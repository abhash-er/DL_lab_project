from defining_attributes import get_att_hierarchy

hier = get_att_hierarchy()


def show_print():
    print(hier.keys())

    print("simple_att")
    print("===========")
    print(len(hier["simple_att"]))
    print(hier["simple_att"][:5])
    print()

    print("clsWtype")
    print("===========")
    print(len(hier["clsWtype"]))
    print(hier["clsWtype"][:5])
    print()

    print("cls2Id")
    print("===========")
    print(len(hier["cls2Id"]))
    print(hier["cls2Id"])
    print()

    print("cls2type")
    print("===========")
    print(len(hier["cls2type"]))
    print(hier["cls2type"])
    print()

    print("att_types")
    print("===========")
    print(len(hier["att_types"]))
    print(hier["att_types"])
    print()

    print("word2attId")
    print("===========")
    print(len(hier["word2attId"]))
    print(hier["word2attId"])
    print()

    print("extended_materials")
    print("===========")
    print(len(hier["extended_materials"]))
    print(hier["extended_materials"])
    print()


show_print()
