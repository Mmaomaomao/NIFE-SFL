import SFLV1_ResNet_HAM10000, SFLV2_ResNet_HAM10000
from ndd_fe import nddf
from typing import List, Dict, Tuple
import pypbc
import charm
from charm.toolbox.integergroup import IntegerGroupQ, integer
from ndd_fe.src.helpers.additive_elgamal import AdditiveElGamal, ElGamalCipher
import random
import hashlib

IntegerGroupElement = charm.core.math.integer.integer
ElGamalKey = Dict[str, IntegerGroupElement]
#生成群
debug = True
p = integer(
    148829018183496626261556856344710600327516732500226144177322012998064772051982752493460332138204351040296264880017943408846937646702376203733370973197019636813306480144595809796154634625021213611577190781215296823124523899584781302512549499802030946698512327294159881907114777803654670044046376468983244647367)
q = integer(
    74414509091748313130778428172355300163758366250113072088661006499032386025991376246730166069102175520148132440008971704423468823351188101866685486598509818406653240072297904898077317312510606805788595390607648411562261949792390651256274749901015473349256163647079940953557388901827335022023188234491622323683)
elgamal_group = IntegerGroupQ()
elgamal = AdditiveElGamal(elgamal_group, p, q)
elgamal_params = {"group": elgamal_group, "p": int(p)}
G2_group = IntegerGroupQ()
G2 = AdditiveElGamal(G2_group, p, q)
G2_params = {"group": G2_group, "p": int(p)}
# 输出群
def output_p():
    return {"group": elgamal_group, "p": int(p)}
# 创建哈希函数H1
def H1(data: str, p: integer) -> integer:
    hash_object = hashlib.sha256(data.encode('utf-8'))
    return integer(int(hash_object.hexdigest(), 16) % p)
# 定义哈希函数 H0: {0, 1}* -> G2
def H0(data: str) -> IntegerGroupElement:
    hash_object = hashlib.sha256(data.encode('utf-8')).hexdigest()
    return elgamal_group.hash(hash_object, G2)
# 定义双线性配对
def bilinear_pairing(P, Q):
    return elgamal.pairing(P, Q)

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
        master_public_key, master_secret_key = nddf.set_up(self.security_parameter, self.vector_length)
        g2 = nddf.G2_group.randomGen()
        self.tk=[]
        self.master_public_key = master_secret_key
        self.master_secret_key = master_secret_key
        self.skagg = integer(random.randint(1, nddf.p-1))
        for i in range(0, self.vector_length-2):
            self.tk[i] = integer(random.randint(1, nddf.p - 1))
        self.vkagg = g2 ** self.skagg

    def run_key_drive(self, master_secret_key):
        sk_convolution, skf = nddf.KeyDerive(master_secret_key, self.master_public_key, self.master_public_key,
                                             self.ctr, self.y, self.aux)

        self.sk_convolution = sk_convolution
        self.skf = skf

    def distributed_training(self):
        pass

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

class User(object):

    # 初始化，设置公共参数
    def __init__(self, security_parameter: int, vector_length: int, ctr: int, y: List[int], aux: str):
        self.security_parameter = security_parameter
        self.vector_length = vector_length
        self.ctr = ctr
        self.y = y
        self.aux = aux

    # 生产公钥和私钥
    def set_up(self):
        master_public_key, master_secret_key = nddf.set_up(self.security_parameter, self.vector_length)
        g2 = nddf.G2_group.randomGen()
        self.tk=[]
        self.master_public_key = master_secret_key
        self.master_secret_key = master_secret_key
        self.skagg = integer(random.randint(1, nddf.p-1))
        for i in range(0, self.vector_length-2):
            self.tk[i] = integer(random.randint(1, nddf.p - 1))
        self.vkagg = g2 ** self.skagg

    def run_key_drive(self, master_secret_key):
        sk_convolution, skf = nddf.KeyDerive(master_secret_key, self.master_public_key, self.master_public_key,
                                             self.ctr, self.y, self.aux)

        self.sk_convolution = sk_convolution
        self.skf = skf

    def distributed_training(self):
        pass

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