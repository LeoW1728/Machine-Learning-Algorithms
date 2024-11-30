from Data import Data
from Item import Item, Melon
from Node import Node, Tree

Melons_data = Data('quality',
['tag', 'color', 'root', 'sound', 'texture', 'belly', 'touch' , 'quality'],
Melon(tag=1, color='green' , root='shrink', sound='blur' , texture='clear' , belly='concave', touch='smooth', quality='good'),
Melon(tag=2, color='black' , root='shrink', sound='deep' , texture='clear' , belly='concave', touch='smooth', quality='good'),
Melon(tag=3, color='black' , root='shrink', sound='blur' , texture='clear' , belly='concave', touch='smooth', quality='good'),
Melon(tag=4, color='green' , root='shrink', sound='deep' , texture='clear' , belly='concave', touch='smooth', quality='good'),
Melon(tag=5, color='white' , root='shrink', sound='blur' , texture='clear' , belly='concave', touch='smooth', quality='good'),
Melon(tag=6, color='green' , root='curved', sound='blur' , texture='clear' , belly='wedged' , touch='slime' , quality='good'),
Melon(tag=7, color='black' , root='curved', sound='blur' , texture='blur'  , belly='wedged' , touch='slime' , quality='good'),
Melon(tag=8, color='black' , root='curved', sound='blur' , texture='clear' , belly='wedged' , touch='smooth', quality='good'),

Melon(tag=9, color='black' , root='curved', sound='deep' , texture='blur'  , belly='wedged' , touch='smooth', quality='bad' ),
Melon(tag=10,color='green' , root='erect' , sound='chip' , texture='clear' , belly='smooth' , touch='slime' , quality='bad' ),
Melon(tag=11,color='white' , root='erect' , sound='chip' , texture='bad'   , belly='smooth' , touch='smooth', quality='bad' ),
Melon(tag=12,color='white' , root='shrink', sound='blur' , texture='bad'   , belly='smooth' , touch='slime' , quality='bad' ),
Melon(tag=13,color='green' , root='curved', sound='blur' , texture='blur'  , belly='concave', touch='smooth', quality='bad' ),
Melon(tag=14,color='white' , root='curved', sound='deep' , texture='blur'  , belly='concave', touch='smooth', quality='bad' ),
Melon(tag=15,color='black' , root='curved', sound='blur' , texture='clear' , belly='wedged' , touch='slime' , quality='bad' ),
Melon(tag=16,color='white' , root='shrink', sound='blur' , texture='bad'   , belly='smooth' , touch='smooth', quality='bad' ),
Melon(tag=17,color='green' , root='shrink', sound='deep' , texture='blur'  , belly='wedged' , touch='smooth', quality='bad' )
)

Same_melon_data = Data('quality',
     ['tag', 'color', 'root'  , 'sound', 'texture', 'belly'  , 'touch' , 'quality'],
    Melon(tag=1, color='green' , root='shrink', sound='blur' , texture='clear' , belly='concave', touch='smooth', quality='good'))

class DecisionTree:
    def __init__(self, feats : list, Data : Data):
        self.data = Data
        F = Data.feats
        for feat in feats:
            if feat not in F:
                self.feats = F
                break
        else:
            self.feats = feats
        self.tree = None

    def update_tree(self):
        for u, v in self.tree.nodes.items():
            if isinstance(v.feat(),str):
                v.set_feat({'node feature':v.feat()})

    def generate_tree(self, data : Data, feats : list) -> Node:
        
        if not data: # Data missing
            return None
        node = Node()

        if not self.tree: # if tree not constructed
            self.tree = Tree(node)
        
        
        # Data extraction with certain feats
        data = data.subdata(feats,{})
        
        A = data.most('')

        reduced_feats = [feat for feat in data.feats if feat != data.main_feat] # exclude the main feature
        
        # if all nodes are in the same class
        if len(reduced_feats) == 1:
            # node.set_feat(data.feats[0])
            node.set_feat(A)
            # print('mode 1 ||' , node.ID(), " set as: ", node.feat())
            return node

        

        # if no feats or all samples have the same value in feats
        Total_H = sum(data.IV(feat) for feat in data.feats)
        if Total_H == 0 or not reduced_feats:
            node.set_feat(A)
            # if not feats:
            #     node.set_feat(A)
            # else:
            #     feat_most, feat_most_count = data.feats[0], 0
            #     for feat in data.feats:
            #         if (count := data.count(feat)) > feat_most_count:
            #             feat_most_count = count
            #             feat_most = feat
            #     node.set_feat(feat_most)
            # print('mode 2 ||' , node.ID(), " set as: ", node.feat())
            return node

        # choose the best splitting feature
        gd = data.gain_decision(feats)
        chosen_feat, gain = gd['decision'], gd['pure_gain'][0][1]
        
        if chosen_feat != data.main_feat and gain > 0:
            # print(gd)
            node.set_feat(chosen_feat)
        else:
            node.set_feat(A)
            return node
        # print('mode 3 ||' , node.ID(), " set as: ", node.feat())
        # print('Chosen: ',chosen_feat)
        chosen_feat_set = self.data.feat_val(chosen_feat)
        partial_feats = [feat for feat in feats if feat != chosen_feat]
        for feat_value in chosen_feat_set:
            # print('\nData: {0}\n\n'.format(len(data)))
            partial_data = data.subdata(partial_feats, {chosen_feat: feat_value})
            if not partial_data:
                node_tag = {'choise':feat_value,'node feature':A}
                kid = Node(feat = node_tag)
            else:
                kid = self.generate_tree(partial_data, partial_feats)
                if isinstance(kid.feat(),str):
                    kid.set_feat({'choise':feat_value,'node feature':kid.feat()})
            self.tree.append(node, kid)
        return node

    def predict(self, item: Item):
        if self.tree:
            regression, now_id = '', self.tree.root.ID()
            while regression not in self.data.main_feat_v:
                now_node = self.tree[now_id]
                feat = now_node.feat()['node feature']
                if feat in self.data.main_feat_v:
                    regression = feat
                    break
                i = item[feat]
                if now_node.kids_ids:
                    for k in now_node.kids_ids:
                        kid = self.tree[k]
                        c = kid.feat()['choise']
                        if c == i:
                            now_id = kid.ID()
            return regression



# print(Melons_data[0])
# m = Melons_data[0]
# for P,M in m:
#     print(P,'=',M)
Melons_data.update()
Melon_Model = DecisionTree([feat for feat in Melons_data.feats if feat != 'tag'], Melons_data)
Melon_Model.generate_tree(Melon_Model.data, Melon_Model.feats)
Melon_Model.update_tree()
Melon_Model.tree.sort()
print(Melon_Model.tree)
print(Melon_Model.predict(Melon(tag=18,color='black' , root='shrink', sound='deep' , texture='blur'  , belly='wedged' , touch='slime' , quality='bad' )))