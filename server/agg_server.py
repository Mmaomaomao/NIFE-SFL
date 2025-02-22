from ndd_fe import nddf
from model.common_param import CommonParam


class AggServer(object):
    def __init__(self, common_param: CommonParam):
        self.common_param = common_param

    def create_key_pair(self):
        master_public_key, master_secret_key = nddf.set_up(self.common_param.security_param, 1, self.common_param.G1)
        aux_public_key_arr, aux_secret_key_arr = nddf.set_up(self.common_param.security_param, 1, self.common_param.G2)
        self.aux_public_key = aux_public_key_arr[0]
        self.aux_secret_key = aux_secret_key_arr[0]
        self.public_key = master_public_key[0]
        self.secret_key = master_secret_key[0]

    def secure_agg(self, kgc_pub_key, skf, C_t, y, T_t_arr):
        kgc_pub_key_arr = [kgc_pub_key]
        self_sec_key_arr = [self.secret_key]
        W_t_1 = nddf.Decrypt(kgc_pub_key_arr, skf, self_sec_key_arr, C_t, y)
        T_t_agg = 1
        for T_t in T_t_arr:
            T_t_agg = T_t_agg * T_t
        T_t_agg = T_t_agg ** self.secret_key
        return W_t_1, T_t_agg
