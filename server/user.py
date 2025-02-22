import random

from ndd_fe import nddf
from model.common_param import CommonParam


class User(object):
    def __init__(self, common_param: CommonParam, data_set):
        self.common_param = common_param

    def create_key_pair(self):
        master_public_key, master_secret_key = nddf.set_up(self.common_param.security_param, 1, self.common_param.G1)
        self.public_key = master_public_key[0]
        self.secret_key = master_secret_key[0]

    def set_tk(self):
        self.tk = random.randint(0, self.common_param.p)

    def train_task(self):
        # todo：函数待补充
        self.W_t = None

    def secure_agg(self, agg_pub_key, ctr, aux):
        agg_pub_key_arr = [agg_pub_key]
        user_pub_key_arr = [self.public_key]
        user_sec_key_arr = [self.secret_key]
        ct_t = nddf.Encrypt(agg_pub_key_arr, user_pub_key_arr, user_sec_key_arr, ctr, self.tk, aux)
        h0_data = nddf.H0(ctr)
        T_t = (h0_data ** self.tk) * (self.common_param.g1 ** self.W_t)
        self.ct_t = ct_t
        self.T_t = T_t
        return ct_t, T_t
