import torch
import mp_lib.torch_utils as torch_utils
from loguru import logger


class xdict(dict):
    def __init__(self, mydict=None):
        if mydict is None:
            return

        for k, v in mydict.items():
            self[k] = v

    def filter(self, keyword):
        out_dict = {}
        for k in self.keys():
            if keyword in k:
                out_dict[k.replace(keyword, "")] = self[k]
        return xdict(out_dict)

    def search(self, keyword):
        out_dict = {}
        for k in self.keys():
            if keyword in k:
                out_dict[k] = self[k]
        return xdict(out_dict)

    def diff_keys(self, keys):
        my_keys = list(self.keys())
        i_have_it_does_not = []
        it_has_i_do_not = []
        for key in my_keys:
            if key not in keys:
                i_have_it_does_not.append(key)

        for key in keys:
            if key not in my_keys:
                it_has_i_do_not.append(key)
        return i_have_it_does_not, it_has_i_do_not

    def rm(self, keyword, keep_list=[], verbose=False):
        out_dict = {}
        for k in self.keys():
            if keyword not in k or k in keep_list:
                out_dict[k] = self[k]
            else:
                if verbose:
                    print(f"Removing: {k}")
        return xdict(out_dict)

    def register(self, k, v):
        assert k not in self.keys()
        self[k] = v

    def overwrite(self, k, v):
        assert k in self.keys()
        self[k] = v

    def add(self, k, v):
        self[k] = v

    def remove(self, k):
        assert k in self.keys()
        self.pop(k, None)

    def merge(self, dict2):
        assert isinstance(dict2, (dict, xdict))
        mykeys = set(self.keys())
        intersect = mykeys.intersection(set(dict2.keys()))
        assert len(intersect) == 0, f"Merge failed: duplicate keys ({intersect})"
        self.update(dict2)

    def mul(self, scalar):
        if isinstance(scalar, int):
            scalar = 1.0 * scalar
        assert isinstance(scalar, float)
        out_dict = {}
        for k in self.keys():
            if isinstance(self[k], list):
                out_dict[k] = [v * scalar for v in self[k]]
            else:
                out_dict[k] = self[k] * scalar
        return xdict(out_dict)

    def prefix(self, text):
        out_dict = {}
        for k in self.keys():
            out_dict[text + k] = self[k]
        return xdict(out_dict)

    def replace_keys(self, str_src, str_tar):
        out_dict = {}
        for k in self.keys():
            old_key = k
            new_key = old_key.replace(str_src, str_tar)
            out_dict[new_key] = self[k]
        return xdict(out_dict)

    def postfix(self, text):
        out_dict = {}
        for k in self.keys():
            out_dict[k + text] = self[k]
        return xdict(out_dict)

    def sorted_keys(self):
        return sorted(list(self.keys()))

    def to(self, dev):
        return xdict(torch_utils.dict2dev(self, dev))

    def to_torch(self):
        return xdict(torch_utils.dict2torch(self))

    def to_np(self):
        return xdict(torch_utils.dict2np(self))

    def tolist(self):
        return xdict(torch_utils.thing2list(self))

    def detach(self):
        out = {}
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                v = v.cpu().detach()
            out[k] = v
        return xdict(out)

    def rm_keywords(self, keywords, keep_list, verbose=False):
        uni = self
        for keyword in keywords:
            uni = uni.rm(keyword, keep_list, verbose)
        return uni

    def has_invalid(self):
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                if torch.isnan(v).any():
                    logger.warning(f"{k} contains nan values")
                    return True
                if torch.isinf(v).any():
                    logger.warning(f"{k} contains inf values")
                    return True
        return False
