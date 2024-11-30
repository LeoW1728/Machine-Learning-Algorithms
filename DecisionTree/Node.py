def register_maker(init_ID):
    now_ID = init_ID
    object_list = {}
    def register(object):
        nonlocal now_ID, object_list
        object_list[now_ID] = object
        now_ID += 1
        return (now_ID - 1)
    return register, object_list

node_register, node_full_list = register_maker(-1)

class Node:
    def __str__(self):
        if self.__id == -1:
            return "<Node object: NULL at {0}>".format(hex(id(self)))
        if self.kids_ids:
            node_str = "<Node object #{0}# at {1}, id = {2}, kids_ids = {3}>".format(
                self.__feat, hex(id(self)), self.__id, self.kids_ids)
        else:
            node_str = "<Node object #{0}# at {1}, id = {2}>".format(
                self.__feat, hex(id(self)), self.__id)
        return node_str

    def __init__(self, *, feat = []):
        self.__feat = feat # feat
        self.kids_ids = []
        # print("Creating node with id %d...\n" % (node_id))
        self.__id = node_register(self)

    def set_feat(self, feat):
        self.__feat = feat

    def feat(self):
        return self.__feat

    def append(self, kid_id):
        # print(self.__id, " appending ", kid_id)
        if kid_id not in self.kids_ids:
            self.kids_ids.append(kid_id)

    def set_ID(self, ID):
        self.__id = ID

    def ID(self):
        return self.__id

    def __setitem__(self, index, value):
        self.kids_ids[index] = value

    def __delitem__(self, index):
        del self.kids_ids[index]

    def __getitem__(self, index):
        return self.kids_ids[index]

    def __iter__(self):
        return iter(self.kids_ids)

    def kids(self) -> list:
        return self.kids_ids
    
Null = Node()

class Tree:
    def __init__(self, Root : Node):
        self.nodes = {}
        self.root = Root
        self.nodes[Root.ID()] = Root
    
    def __str__(self):
        tree_str = "<Tree object at {0}, nodes are ".format(hex(id(self))) + '{'
        for i, n in self.nodes.items():
            tree_str += '\n' + repr(str(i))+": "+ str(n) +","
        tree_str += "\b}\n>"
        return tree_str

    def __delitem__(self, ID):
        del self.nodes[ID]

    def __getitem__(self, ID):
        return self.nodes[ID]

    def __setitem__(self, ID, node):
        self.nodes[ID] = node

    def __iter__(self):
        return iter(self.nodes.items())

    def append(self, parent : Node, *appending_nodes):
        for node in appending_nodes:
            i = node.ID()
            parent.append(i)
            self[i] = node
    
    def parent_node(self, kid : Node):
        if kid.ID() not in self.nodes:
            return Null
        i = kid.ID()
        for t, node in iter(self):
            if i in node.kids():
                i = node.ID()
                break
        else:
            return Null
        return self[i]
            

    def kids(self, node : Node):
        if not node.kids_ids:
            return iter([self[kid_id] for kid_id in node.kids_ids])
        else:
            return None

    def root(self):
        return self.root

    def sort(self):
        self.nodes = dict(sorted(self.nodes.items(), key=lambda x:x[0]))


# Node1 = Node()
# Node2 = Node()
# Node3 = Node()
# NewTree = Tree(Root)
# NewTree2 = Tree(Node1)
# NewTree.append(Root, Node1, Node2)
# NewTree.append(Node1, Node3)
# NewTree2.append(Node1, Node3)
# print(NewTree2)
# print(NewTree2.parent_node(Node3))