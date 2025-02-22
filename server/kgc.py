from ndd_fe import nddf
from model.common_param import CommonParam


class KGC(object):
    def __init__(self, common_param: CommonParam):
        self.common_param = common_param

    def create_key_pair(self):
        master_public_key, master_secret_key = nddf.set_up(self.common_param.security_param, 1, self.common_param.G1)
        self.public_key = master_public_key[0]
        self.secret_key = master_secret_key[0]

    def run_key_drive(self, master_secret_key,ctr):
        sk_convolution, skf = nddf.KeyDerive(master_secret_key, self.master_public_key, self.master_public_key,
                                             ctr, self.y, self.aux)

        self.sk_convolution = sk_convolution
        self.skf = skf

    def key_drive(self, users_master_public_key, users_master_secret_key,ctr, y, aux):
        secret_key = [self.secret_key]
        skf = nddf.KeyDerive(secret_key, users_master_public_key, users_master_secret_key, ctr, y,
                             aux)
        return skf

