from nife_sfl_project import SFLV1_ResNet_HAM10000, SFLV2_ResNet_HAM10000
from nife_sfl_project.ndd_fe import nddf
from typing import List, Dict, Tuple
import pypbc


class MainExecProcess(object):

    # 初始化，设置公共参数
    def __init__(self, security_parameter: int, vector_length: int, ctr: int, y: List[int], aux: str):
        self.security_parameter = security_parameter
        self.vector_length = vector_length
        self.ctr = ctr
        self.y = y
        self.aux = aux

    # 生产公钥和私钥
    def set_up(self):
        master_public_key, master_secret_key, vkagg = nddf.set_up(self.security_parameter, self.vector_length)
        self.master_public_key = master_secret_key
        self.master_secret_key = master_secret_key
        self.vkagg = vkagg

    def run_key_drive(self, master_secret_key):
        sk_convolution, skf = nddf.KeyDerive(master_secret_key, self.master_public_key, self.master_public_key,
                                             self.ctr, self.y, self.aux)

        self.sk_convolution = sk_convolution
        self.skf = skf

    def distributed_training(self):
        self.Tt = 2
        self.vkt = 1
        self.wk_vec = []

    def secure_aggregation(self, master_secret_key2, master_public_key3, master_secret_key3, X):
        cipher = nddf.Encrypt(self.master_public_key, master_secret_key2, master_public_key3, self.ctr, X, self.aux)
        text = nddf.Decrypt(self.master_public_key, self.skf, master_secret_key3, cipher, self.y)

        self.cipher = cipher
        self.text = text

    def verification(self):
        pairing = pypbc.Pairing
        return pairing(self.Tt, nddf.elgamal_group.g1) == pairing(nddf.H0(self.ctr), self.vkt) * pairing(self.vkagg,
                                                                                                         self.wk_vec[0])

    def exec_all(self, master_secret_key, master_secret_key2, master_public_key3, master_secret_key3, X):
        self.set_up()
        self.run_key_drive(master_secret_key)
        self.distributed_training()
        self.secure_aggregation(master_secret_key2, master_public_key3, master_secret_key3, X)
        self.verification()
