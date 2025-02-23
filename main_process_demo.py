import random,copy
from model.common_param import CommonParam
from server import user, agg_server, kgc, main_server
from fe import fe_function
from SFLV1_ResNet_HAM10000 import Client,device,ResNet18_client_side
# from src.helpers import dummy_discrete_log, get_int, get_modulus
import torch
from server.main_server import ResNet18_server_side, Baseblock, train_server, evaluate_server

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    epochs = 200
    lr = 0.0001
    tk_filter_tank = []
    process_common_param = CommonParam(fe_function.input_G1, fe_function.input_G2, fe_function.input_GT, fe_function.p, fe_function.g1_generator,
                                       fe_function.g2_generator, fe_function.security_param)
    for t in range(0, epochs):
        user_arr = []
        ctr = 222
        y = []
        ct_t_arr = []
        T_t_arr = []
        VK_t_arr = []
        users_pub_key_arr = []
        w_locals_client = []
        aux = None
        tk_arr = generator_n_dif_random(n, fe_function.p)
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
        main_server_ele = main_server.MainServer(process_common_param)
        skf = kgc_ele.key_drive(users_pub_key_arr, ctr, y, aux)
        net_glob_client = ResNet18_client_side()
        net_glob_client.to(device)
        for single_user in user_arr:
            single_client = Client(net_glob_client,single_user, lr, device)
            single_user.train_task(single_client)
            w_client = main_server_ele.train_task(single_user.fx_client, single_user.y, epochs, t, single_user, aux, ell = iter)
            w_locals_client.append(copy.deepcopy(w_client))
            ct_t, T_t, VK_t = single_user.secure_agg(kgc_ele.public_key, agg_server_ele.public_key, ctr, aux,
                                                     agg_server_ele.aux_public_key)
            ct_t_arr.append(ct_t)
            T_t_arr.append(T_t)
            VK_t_arr.append(VK_t)

        W_t_1, T_t_agg = agg_server_ele.secure_agg(kgc_ele.public_key, skf, ct_t_arr, y, T_t_arr)
        VK_t_agg = kgc_ele.secure_agg(VK_t_arr)
        verify_res = fe_function.verify(T_t_agg, fe_function.g2_generator, ctr, VK_t_agg, fe_function.g1_generator, n, W_t_1,
                                 agg_server_ele.aux_public_key)
        if not verify_res:
            break
        else:
            net_glob_client.load_state_dict(W_t_1)
