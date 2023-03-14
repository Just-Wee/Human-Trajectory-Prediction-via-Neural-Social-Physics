"""
Occupy GPU
"""
import argparse
import os
import time
import psutil

import torch


def get_gpu_mem_info(gpu_id=0):
    """
    根据显卡 id 获取显存使用信息, 单位 MB
    :param gpu_id: 显卡 ID
    :return: total 所有的显存，used 当前使用的显存, free 可使用的显存
    """
    import pynvml
    pynvml.nvmlInit()
    if gpu_id < 0 or gpu_id >= pynvml.nvmlDeviceGetCount():
        print(r'gpu_id {} 对应的显卡不存在!'.format(gpu_id))
        return 0, 0, 0

    handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
    total = round(meminfo.total / 1024 / 1024, 2)
    used = round(meminfo.used / 1024 / 1024, 2)
    free = round(meminfo.free / 1024 / 1024, 2)
    return total, used, free


def get_cpu_mem_info():
    """
    获取当前机器的内存信息, 单位 MB
    :return: mem_total 当前机器所有的内存 mem_free 当前机器可用的内存 mem_process_used 当前进程使用的内存
    """
    mem_total = round(psutil.virtual_memory().total / 1024 / 1024, 2)
    mem_free = round(psutil.virtual_memory().available / 1024 / 1024, 2)
    mem_process_used = round(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024, 2)
    return mem_total, mem_free, mem_process_used

def get_memory(device_ids=[0], rate=0.9):
    mat_list = [[] for _ in range(len(device_ids))]
    while True:
        total, used, _ = get_gpu_mem_info(0)
        if used / total > rate:
            break
        try:
            for i, d_id in enumerate(device_ids):
                t_mat = torch.ones((1024, 1024), dtype=torch.float32).to(device='cuda:%d'%(d_id))
                mat_list[i].append(t_mat)
        except:
            break
    assert all(mat_list)
    return mat_list

def run_gpu(mat_list):
    t0 = time.time()
    cnt = 0
    while True:
        for d_mat_list in mat_list:
            for mat in d_mat_list:
                mat *= d_mat_list[0]
        cnt += 1
        if cnt % 60 == 0:
            print(time.strftime('Runing %y-%m-%d %H:%m', time.localtime()))
        if cnt // 60 >= 60 * 8:
            t1 = time.time() - t0
            th = t1 // (60 * 60)
            tm = t1 % (60 * 60) // 60
            print('Occupy %d hours %d minutes!' % (th, tm))
            cnt = 0
        time.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NSP')
    parser.add_argument('--gpu_index', '-gi', type=int, nargs='+', default=[0])
    args = parser.parse_args()
    print('Occupy GPU:', args.gpu_index)

    mat_list = get_memory(args.gpu_index)
    print('Allocate cache success! Allocate matrix num: %d' %
          (sum([len(m_l) for m_l in mat_list])))
    run_gpu(mat_list)
