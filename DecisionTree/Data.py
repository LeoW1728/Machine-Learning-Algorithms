from math import log2

class Data:

    def feat_val(self, feat):
        feat_set = set([item[feat] for item in self.items]) # different values for items' feature
        return feat_set

    def main_feat_val(self):
        self.main_feat_v = list(set([item[self.main_feat] for item in self.items])) # different values for items' main feature

    def entropy(self): # calculate an entropy
        subdatav = self.subdata_values(self.main_feat)
        subdatav_set = set(subdatav)
        probs, l = [], len(subdatav)
        for p in subdatav_set:
            probs.append(subdatav.count(p) / l)
        H = sum(-prob * log2(prob) for prob in probs if prob != 0)
        return H

    def update(self):
        self.main_feat_val()
        self.__H = self.entropy()

    def __init__(self, main_feat : str, feats: list, *items):
        if main_feat not in feats:
            raise ValueError("Main feat must be included in feats.")
        self.main_feat = main_feat
        self.feats = feats
        if items:
            self.items = list(items)
        else:
            self.items = []

    def append(self, item):
        for property, value in item.properties.items():
            if property not in self.feats:
                self.feats.append(property)
        self.items.append(item)

    def __len__(self):
        return len(self.items)
    
    def __setitem__(self, index, item):
        self.items[index] = item
    
    def __getitem__(self, index):
        return self.items[index]

    def __delitem__(self, index):
        del self.items[index]

    def __iter__(self):
        return iter(self.items)

    def __str__(self):
        datastr = '<Data object at {0}, data: [\n'.format(hex(id(self)))
        for datum in self.items:
            datastr += '| ' + str(datum) + '\n'
        datastr += ']>'
        return datastr

    def subdata_values(self, feat, *, mode = 'normal'):
        subdatav = []
        if mode == 'normal':
            for item in self.items:
                if item[feat]:
                    subdatav.append(item[feat])
            return subdatav
        elif mode == 'append':
            for item in self.items:
                if item[feat]:
                    subdatav.append((item[self.main_feat], item[feat]))
            return subdatav

    def count(self, feat : str) -> str:
        if feat not in self.feats:
            return 0
        count = 0
        for item in self.items:
            if item[feat]:
                count += 1
        return count

    def most(self, feat : str) -> str:
        if not feat:
            feat = self.main_feat
            most_feat_value, most_count, count = '', 0, 0
            for main_feat_value in self.main_feat_v:
                for item in self.items:
                    if item[feat] == main_feat_value:
                        count += 1
                if count >= most_count:
                    most_feat_value = main_feat_value
                    most_count = count
            return most_feat_value
        else:
            most_feat_value, most_count, count= '', 0, 0
            for feat_value in self.feat_val(feat):
                for item in self.items:
                    if item[feat] == feat_value:
                        count += 1
                if count >= most_count:
                    most_feat_value = feat_value
                    most_count = count
            return most_feat_value

    def subdata(self, feats : list, conditions : dict):
        subdata = Data(self.main_feat, feats)
        if conditions:
            for datum in self.items:
                for condition, cvalue in conditions.items():
                    if datum[condition] != cvalue:
                        break
                else:
                    subdata.append(datum.copy(feats))
        else:
            for datum in self.items:
                subdata.append(datum.copy(feats))
        subdata.update()
        return subdata

    def gain(self, feat) -> float: # calculate a gain wrt a feature
        if feat == self.main_feat:
            return 0.
        else:
            subdatav = self.subdata_values(feat, mode = 'append')
            if not subdatav:
                return None
            feat_set = self.feat_val(feat)
            D, feat_entropies = len(self), []
            for feat_value in feat_set: # for each possible feature's value calculate different entropy
                feat_subdata = [value[0] for value in subdatav if value[1]==feat_value]
                D_feat = len(feat_subdata)
                main_probs = [feat_subdata.count(main_feat_value)/D_feat for main_feat_value in self.main_feat_v]
                H = -sum(p * log2(p) for p in main_probs if p != 0)
                feat_entropies.append((H,D_feat/D))
            G = -sum(H*P for H, P in feat_entropies)
            return G + self.__H

    def gain_decision(self, feats : list):
        F = feats
        if not feats:
            F = self.feats
        gains = {feat:self.gain(feat) for feat in F if feat in self.feats}
        gains_sorted = sorted(gains.items(), key=lambda x:x[1], reverse=True)
        res = {'decision': gains_sorted[0][0], 'pure_gain':gains_sorted}
        return res

    def IV(self, feat): # intrinsic value
        if feat == self.main_feat:
            return self.__H
        subdatav = self.subdata_values(feat)
        if not subdatav:
            return None
        subdatav_set = set(subdatav)
        probs, l = [], len(subdatav)
        for p in subdatav_set:
            probs.append(subdatav.count(p) / l)
        iv = sum(-prob * log2(prob) for prob in probs if prob != 0)
        return iv

    def gain_ratio(self, feat): # gain ratio
        if feat == self.main_feat:
            return 0
        else:
            subdatav = self.subdata_values(feat, mode = 'append')
            if not subdatav:
                return None
            feat_set = self.feat_val(feat)
            D, feat_entropies = len(self), []
            for feat_value in feat_set: # for each possible feature's value calculate different entropy
                feat_subdata = [value[0] for value in subdatav if value[1]==feat_value]
                D_feat = len(feat_subdata)
                main_probs = [feat_subdata.count(main_feat_value)/D_feat for main_feat_value in self.main_feat_val]
                H = -sum(p * log2(p) for p in main_probs if p != 0)
                feat_entropies.append((H,D_feat/D))
            G = -sum(H*P for H, P in feat_entropies)
            iv = -sum(P * log2(P) for H, P in feat_entropies if P != 0)
            return (G + self.__H) / iv