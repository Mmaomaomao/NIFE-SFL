import random

from model.common_param import CommonParam
from server import user, agg_server, kgc, main_server
from ndd_fe import nddf


# from src.helpers import dummy_discrete_log, get_int, get_modulus

# 生成群

def generator_n_dif_random(n, p):
    numbers = []
    while True:
        number = random.randint(0, p)
        if number not in numbers:
            numbers.append(number)
        if len(numbers) == n:
            break


if __name__ == '__main__':
    n = 5
    T = 10
    tk_filter_tank = []
    process_common_param = CommonParam(nddf.input_G1, nddf.input_G2, nddf.input_GT, nddf.p, nddf.g1_generator,
                                       nddf.g2_generator, nddf.security_param)

    for t in range(0, T):
        user_arr = []
        ctr = 222
        y = []
        ct_t_arr = []
        T_t_arr = []
        VK_t_arr = []
        users_pub_key_arr = []
        aux = None
        tk_arr = generator_n_dif_random(n, nddf.p)
        for i in range(0, n):
            data_set = None
            user_ele = user.User(process_common_param, data_set)
            user_ele.create_key_pair()
            user_ele.set_tk(tk_arr[i])
            users_pub_key_arr.append(user_ele.public_key)
            user_arr.append(user_ele)
            y.append(1 / n)
        agg_server_ele = agg_server.AggServer(process_common_param)
        agg_server_ele.create_key_pair()
        kgc_ele = kgc.KGC(process_common_param)
        kgc_ele.create_key_pair()
        skf = kgc_ele.key_drive(users_pub_key_arr, ctr, y, aux)
        for single_user in user_arr:
            single_user.train_tasktrain_task()
            # todo:补充训练代码
            ct_t, T_t, VK_t = single_user.secure_agg(kgc_ele.public_key, agg_server_ele.public_key, ctr, aux,
                                                     agg_server_ele.aux_public_key)
            ct_t_arr.append(ct_t)
            T_t_arr.append(T_t)
            VK_t_arr.append(VK_t)

        W_t_1, T_t_agg = agg_server_ele.secure_agg(kgc_ele.public_key, skf, ct_t_arr, y, T_t_arr)
        VK_t_agg = kgc_ele.secure_agg(VK_t_arr)
        verify_res = nddf.verify(T_t_agg, nddf.g2_generator, ctr, VK_t_agg, nddf.g1_generator, n, W_t_1,
                                 agg_server_ele.aux_public_key)
        if not verify_res:
            break
