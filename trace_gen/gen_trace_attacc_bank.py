import argparse
import math
import copy
import numpy as np

model = "gpt-3-175B"

dhead = 128
max_L = 2048
data_size = 16 # FP 16

n_attacc = 8
max_n_hbm = 8
n_hbm = 5
n_channel = 16
n_pch = 2
n_rank = 2
n_bank = 4
n_bg = 4
n_row = pow(2, 14)
n_col = pow(2, 5)
prefetch_size = 32 # byte
n_mac = 16


# Granularity size
HBM_GS = {}
HBM_GS['col']     = prefetch_size
HBM_GS['row']     = n_col * HBM_GS['col']
HBM_GS['ba']      = n_row * HBM_GS['row']
HBM_GS['bg']      = n_bank * HBM_GS['ba']
HBM_GS['rank']     = n_bg * HBM_GS['bg']
HBM_GS['pch']     = n_rank * HBM_GS['rank']
HBM_GS['ch']      = n_pch * HBM_GS['pch']
HBM_GS['hbm']     = n_channel * HBM_GS['ch']
HBM_GS['attacc']  = max_n_hbm * HBM_GS['hbm']   # 似乎这个在这里没什么用


## --------------------------------------  HBM memory space -----------------------------------------##
## ------|  legacy CH  |  pCH  |  rank  | BG | BA |  row index  |  column index  |  access granularity  |------ ##
## bits  |     4       |   1   |   1   | 2  | 2  |     14      |        5       |          5           |       ##


## ----------------------------  Commands -------------------------------##
## MACAB: 8tCK (tCCDLx 2)
##  WRGB: 4tCK (write to SRAM not DRAM) write to GEMV buffer
##  MVSB: 4tCK   move vector to softmax buffer
##  MVGB: 4tCK   move vector to gemv buffer unkwon difference to WRGB
##  SFM: 16tCK (for L = 256)  softmax operation

cmd_score_wrgb   = []
cmd_score_mac    = []
cmd_score_mvsb   = []
cmd_sfm          = []
cmd_context_mvgb  = []
cmd_context_mac  = []
cmd_context_mvsb = []

valid_channels = []


def get_list_shape(lst):
    if not isinstance(lst, list):
        return ()

    shape = [len(lst)]
    while isinstance(lst[0], list):
        lst = lst[0]
        shape.append(len(lst))

    return tuple(shape)

def get_all_shape():
    all_commands = {
        'cmd_score_wrgb':cmd_score_wrgb,
        'cmd_score_mac':cmd_score_mac,
        'cmd_score_mvsb':cmd_score_mvsb,
        'cmd_sfm':cmd_sfm,
        'cmd_context_mvgb':cmd_context_mvgb,
        'cmd_context_mac':cmd_context_mac,
        'cmd_context_mvsb':cmd_context_mvsb
    }

    for name,command in all_commands.items():
        print(name,get_list_shape(command))


def cmd_list_reset():
    cmd_score_wrgb   = []
    cmd_score_mac    = []
    cmd_score_mvsb   = []
    cmd_sfm          = []
    cmd_context_mvgb = []
    cmd_context_mac  = []
    cmd_context_mvsb = []

    valid_channel = []

def Attention(L, key_addr, val_addr, itr, valid_channel = n_channel):
    cmd_score_wrgb.append([])
    cmd_score_mac.append([])
    cmd_score_mvsb.append([])
    cmd_sfm.append([])
    cmd_context_mvgb.append([])
    cmd_context_mac.append([])
    cmd_context_mvsb.append([])

    valid_channels.append(valid_channel);

    def score_cpvec(addr_offset, L):
        ## (pCH) C, C, R, R (MAC)
        ## write input vector to gemv buffer
        # number of partition = (R parallel units)

        # Data broadcasting for pch, rank, bg, and ba
        for ba_idx in range(n_bank): # number of partitions
            for col_idx in range(math.ceil(dhead / n_bank / n_mac)):  # 一次写 256 bit
                for lch in range(math.ceil(valid_channel)):
                    # GEMV buffer address, col granularity = 1
                    addr = addr_offset + lch * HBM_GS['ch'] + ba_idx * HBM_GS['ba'] + col_idx
                    hex_addr = hex(addr)[2:]
                    cmd_score_wrgb[itr].append("PIM_WR_GB 0x{0:0>8}".format(hex_addr))

    def score_mac(addr_offset, L):
        ## (pCH) C, C, R, R (MAC)
        # MAC and move output vector to softmax buffer
        ## Vector (1 x k) x Matrix (k x n) multiplication
        ## GEMV unit = adder tree mode
        for n_idx in range(math.ceil(L / n_pch / n_rank / n_bg)):# 16  # 这个循环是128次, 表示每个Bank Group 分到的计算量,一共128列
            cmd_score_mac[itr].append([])
            for k_idx in range(math.ceil(dhead / n_bank / n_mac)): # 2  # 这个循环是2次, 表示每个 Bank-> 每个 MAC 单元分到的计算  时序上
                # 也就是每个head,对于一个channel而言, 需要128组 BG 控制指令, BG内部只需要2条指令就能完成 控制
                idx = k_idx + n_idx * math.ceil(dhead / n_bank / n_mac)

                # All bank command (legacy channel)
                for lch in range(math.ceil(valid_channel)):  # 这个是不同channel 之间, 可以先忽略
                    addr = addr_offset + lch * HBM_GS['ch'] + idx * HBM_GS['col']
                    hex_addr = hex(addr)[2:]
                    cmd_score_mac[itr][-1].append("PIM_MAC_AB 0x{0:0>8}".format(hex_addr))   # 4x128x32
                ## parallelization

            ## MVSB command (Move to Softmax buffer)
            ## A output element is generated for every n_idx
            if n_idx % 16 == 15 or n_idx == math.ceil(L / n_pch / n_rank / n_bg) - 1:  # 为啥是每16个汇集一下? 之间的结果会存储到哪里呢?
                # 合理猜测是每计算16列, 正好出16个结果,然后移动到Softmax buffer中进行softmax操作
                cmd_score_mvsb[itr].append([])
                for bg_idx in range(n_bg): # 以bank group为单位,因为运算的时候也是按bank group为单位的
                    for rank in range(n_rank): # 为啥这里没有pCH? 好奇怪
                        for lch in range(math.ceil(valid_channel)):
                            bank_addr = addr_offset + lch * HBM_GS['ch'] + rank * HBM_GS['rank'] + \
                                        bg_idx * HBM_GS['bg']
                            hex_addr = hex(bank_addr)[2:]
                            cmd_score_mvsb[itr][-1].append("PIM_MV_SB 0x{0:0>8}".format(hex_addr))

    ## (pCH) R, R, C, C (MAC)
    def context_cpvec(addr_offset, L):
        ## write input vector to gemv buffer
        ## number of partition = (BG and BA banks)

        # Data broadcasting for bg and ba
        for rank in range(n_rank):
            for bg_idx in range(n_bg):
                for col_idx in range(math.ceil(L / (n_pch * n_rank * n_bg * n_mac))): # 最后这个n_mac 实际上就是对应的一个256bit, 对齐长度
                    # number of columns of partition = L / (R parallel units)
                    for lch in range(math.ceil(valid_channel)):
                        # GEMV buffer address, col granularity = 1
                        addr = addr_offset + lch * HBM_GS['ch'] + rank * HBM_GS['rank'] + \
                               bg_idx * HBM_GS['bg'] + col_idx
                        hex_addr = hex(addr)[2:]
                        cmd_context_mvgb[itr].append("PIM_MV_GB 0x{0:0>8}".format(hex_addr))

    def context_mac(addr_offset, L):
        # MAC and move output vector to softmax buffer
        ## Vector (1xk) x Matrix (k x n ) multiplication
        ## GEMV unit = mac mode
        for n_idx in range(math.ceil(dhead / (n_bank * n_mac))):
            cmd_context_mac[itr].append([])
            for k_idx in range(math.ceil(L / (n_pch * n_rank * n_bg))):
                idx = k_idx + n_idx * math.ceil(L / (n_pch * n_rank * n_bg))
                for lch in range(math.ceil(valid_channel)):
                    addr = addr_offset + lch * HBM_GS['ch'] + idx * HBM_GS['col']
                    hex_addr = hex(addr)[2:]
                    cmd_context_mac[itr][-1].append("PIM_MAC_AB 0x{0:0>8}".format(hex_addr))

            ## parallelization. Generate 16 elements per n_idx
            cmd_context_mvsb[itr].append([]) # BG 之间是不是能直接累加?
            for ba_idx in range(n_bank):
                for rank in range(n_rank):
                    for lch in range(math.ceil(valid_channel)):
                        bank_addr = addr_offset + lch * HBM_GS['ch'] + rank * HBM_GS['rank'] + \
                                    ba_idx * HBM_GS['ba']
                        hex_addr = hex(bank_addr)[2:]
                        cmd_context_mvsb[itr][-1].append("PIM_MV_SB 0x{0:0>8}".format(hex_addr))

    def softmax(L):
        for lch in range(math.ceil(valid_channel)):
            addr = lch * HBM_GS['ch']
            hex_addr = hex(addr)[2:]
            cmd_sfm[itr].append("PIM_SFM 0x{0:0>8}".format(hex_addr))

    score_cpvec(key_addr, L)

    score_mac(key_addr, L)

    softmax(L)

    context_cpvec(val_addr, L)

    context_mac(val_addr, L)


# n_head and n_req = n_req per a HBM 
def run_attention(dhead, n_head_per_hbm, L, trace_file_name):
    partition_size = math.ceil(max_L * dhead / (n_pch * n_rank * n_bg * n_bank)) # 根据最长的情况,给bank分配空间 , 每个bank需要多少空间
    head_offset = partition_size
    v_offset = pow(2, 23)


    cmd_list_reset()
    ##-- Generate Commands --##
    # 这里生成的command只有计算部分的内容,可能没有同步,同时没有调度,不知道先后顺序
    # 其实看起来还是按照channel对head进行的划分,每个channel都是跑一个head
    num_itr = math.ceil(n_head_per_hbm / (n_channel))
    for itr in range(num_itr):
        remainder = 0
        if (n_head_per_hbm / ((itr+1) * n_channel) < 1):
            remainder = n_head_per_hbm % n_channel
        key_addr = itr * partition_size
        val_addr = key_addr + v_offset
        if remainder == 0:
            Attention(L, key_addr, val_addr, itr)
        else:
            Attention(L, key_addr, val_addr, itr, remainder) # 最后的情况,可能head数量不够,channel用不满


    ##-- Ovelapping Commands --##
    # Barrier机制,方便流水下同步,每个流水级跑完,就来一下
    # 针对每个channel而言的,每个channel都停顿卡一下
    barrier = []
    for lch in range(n_channel):
        addr = lch * HBM_GS['ch']
        hex_addr = hex(addr)[2:]
        barrier.append("PIM_BARRIER 0x{0:0>8}".format(hex_addr))

    # 这里决定指令调度的顺序,把前面的指令聚合起来
    # 这里是以两个head
    total_cmd = []
    for i in range(0, num_itr -1, 2):

        # Head0: Score
        ## WRGB
        # 盲猜这个是从Controller写数据到GEMV buffer
        total_cmd += cmd_score_wrgb[i] # 一个channel 执行4个head, 每个head 占一份
        ## dummy MAC
        # 再写入执行指令
        # 这个更像是流水之前的准备,有点迷
        if i == 0:
            for j in range(valid_channels[i]):
                total_cmd.append(cmd_score_mac[i][0][j])
            ## BARRIER
        total_cmd += barrier

        # 这里就开始流水了,算head 0 , 读入 head 1
        length = math.ceil(L/n_pch/n_rank/n_bg/16) # 将L按Bank Group 拆分, 每个bank Group分的length长度, 但是最后面又有个16, 是不是和stride相互配合使用的?
        for j in range(0, length+1):
            ## MAC (Head0)
            if not j == length: # 不是最后一个循环
                stride = 16;  # 这里的stride合理怀疑是一次运算的对于K的列数 128x16 的 K
                for k in range(stride):
                    if (j*stride+k) >= len(cmd_score_mac[i]): # 总这里看stride似乎是和 K的列有关系, len(cmd_score_mac[i]) 为 128 ,对应一个BG执行的列数
                        break;
                    total_cmd += cmd_score_mac[i][j*stride+k] # 一次加32条指令 16x2
            ## MVSB (Head0)
            if not j == 0:  # 将前面算出的结果移动到Softmax Buffer中
                total_cmd += cmd_score_mvsb[i][j-1] # j-1 正好对应上一次运算出的结果, 和 MVSB生成的顺序也有关系
            ## WRGB (Head1)
            if not j == length: # 不是最后一个循环
                stride = int(n_bank*math.ceil(dhead /n_bank /n_mac)*math.ceil(valid_channels[i+1])/length);
                for k in range(stride):
                    if (j*stride+k) >= len(cmd_score_wrgb[i+1]):
                        break;
                    total_cmd.append(cmd_score_wrgb[i+1][j*stride + k])
            ## BARRIER
            if not j == length:
                total_cmd += barrier

        # Head0: SoftMax, Head1: Score
        length = math.ceil(L/n_pch/n_rank/n_bg/16)
        for j in range(0, length+1):
            ## MAC (Head1)
            if not j == length:
                stride = 16;
                for k in range(stride):
                    if (j*stride+k) >= len(cmd_score_mac[i+1]):
                        break;
                    total_cmd += cmd_score_mac[i+1][j*stride+k]
            ## MVSB (Head1)
            if not j == 0:
                total_cmd += cmd_score_mvsb[i+1][j-1]
            ## SFM (Head0)
            if j == 0:
                total_cmd += cmd_sfm[i] # 一个channel 一条
            ## MVGB (Head0)
            if not j == length:
                if j >= math.floor(length/2): # 为了等一下结果吗?
                    stride = int(n_rank*n_bg*math.ceil(L/(n_pch*n_rank*n_bg*n_mac))*math.ceil(valid_channels[i])/math.ceil(length/2));
                    for k in range(stride):
                        if ((j-math.floor(length/2))*stride + k) >= len(cmd_context_mvgb[i]):
                            break;
                        total_cmd.append(cmd_context_mvgb[i][(j-math.floor(length/2))*stride + k])
            ## BARRIER
            if not j == length:
                total_cmd += barrier

        # Head0: Context, Head1: Softmax
        length = math.ceil(dhead/n_bank/n_mac)
        for j in range(0, length+1):
            ## MAC (Head0)
            if not j == length:
                total_cmd += cmd_context_mac[i][j]
            ## MVSB (Head0)
            if not j == 0:
                total_cmd += cmd_context_mvsb[i][j-1]
            ## SFM (Head1)
            if j == 0:
                total_cmd += cmd_sfm[i+1]
            ## MVGB (Head1)
            if not j == length:
                if j >= math.floor(length/2):
                    stride = int(n_rank*n_bg*math.ceil(L/(n_pch*n_rank*n_bg*n_mac))*math.ceil(valid_channels[i+1])/math.ceil(length/2));
                    for k in range(stride):
                        if ((j-math.floor(length/2))*stride + k) >= len(cmd_context_mvgb[i+1]):
                            break;
                        total_cmd.append(cmd_context_mvgb[i+1][(j-math.floor(length/2))*stride + k])
            ## BARRIER
            if not j == length:
                total_cmd += barrier

        # Head1: Context
        length = math.ceil(dhead/n_bank/n_mac)
        for j in range(0, length+1):
            # 这一段貌似写错了，应该是head-1的，但是貌似影响不大
            ## MAC (Head0)
            if not j == length:
                total_cmd += cmd_context_mac[i][j]
            ## MVSB (Head0)
            if not j == 0:
                total_cmd += cmd_context_mvsb[i][j-1]
            ## BARRIER
            if not j == length:
                total_cmd += barrier


    if num_itr % 2 != 0:
        i = num_itr - 1

        # Score
        ## WRGB
        total_cmd += cmd_score_wrgb[i]
        ## BARRIER
        total_cmd += barrier

        length = math.ceil(L/n_pch/n_rank/n_bg/16)
        for j in range(0, length+1):
            ## MAC
            if not j == length:
                stride = 16;
                for k in range(stride):
                    if (j*stride+k) >= len(cmd_score_mac[i]):
                        break;
                    total_cmd += cmd_score_mac[i][j*stride+k]
            ## MVSB
            if not j == 0:
                total_cmd += cmd_score_mvsb[i][j-1]
            ## BARRIER
            if not j == length:
                total_cmd += barrier

        # SoftMax
        ## SFM (Head0)
        total_cmd += cmd_sfm[i]
        ## MVGB (Head0)
        total_cmd += cmd_context_mvgb[i]
        ## BARRIER
        total_cmd += barrier

        # Context
        length = math.ceil(dhead/n_bank/n_mac)
        for j in range(0, length+1):
            ## MAC
            if not j == length:
                total_cmd += cmd_context_mac[i][j]
            ## MVSB
            if not j == 0:
                total_cmd += cmd_context_mvsb[i][j-1]
            ## BARRIER
            if not j == length:
                total_cmd += barrier


    trace_file = open(trace_file_name, 'w')
    for cmd in total_cmd:
        trace_file.write(cmd + "\n")

    trace_file.close()

def main():
    global dhead, max_L, data_size, n_mac


    parser = argparse.ArgumentParser(description="Output path and operation infos",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-dh", "--dhead", type=int, default=128,
                        help="dhead, default= 128")
    parser.add_argument("-nh", "--nhead", type=int, default=64,
                        help="Number of heads, default=64")
    parser.add_argument("-l", "--seqlen", type=int, default=2048,
                        help="Sequence length L, default= 2048")
    parser.add_argument("-maxl", "--maxlen", type=int, default=4096,
                        help="maximum L, default= 4096")
    parser.add_argument("-db", "--dbyte", type=int, default=2,
                        help="data type (B), default= 2")
    parser.add_argument("-o", "--output", type=str, default="attacc_bank.trace",
                        help="output path")

    args = parser.parse_args()

    dhead = args.dhead
    max_L = args.maxlen
    L = args.seqlen
    n_head_per_hbm = args.nhead

    data_size = args.dbyte
    n_mac = int(HBM_GS['col'] / data_size)

    print("------   Make a trace of bank-level AttAcc   ------")

    args_dict = vars(args)
    print("All Arguments:")
    for key, value in args_dict.items():
        print(f"     {key}: {value}")
    print("---------------------------------------------------")
    run_attention(dhead, n_head_per_hbm, L, args.output)



if __name__ == "__main__":
    main()
