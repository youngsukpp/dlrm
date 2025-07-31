import numpy as np
import math
import random
import pickle
import os
import dlrm_data_pytorch as dp
import sys
import cache_simulator as CacheSimulator
sys.path.insert(0, '..')
from proactivePIM_translation import ProactivePIMTranslation
from multi_hot import Multihot
from itertools import chain
import argparse
from enum import Enum

addr_map = {
    "rank"      : 0,
    "row"       : 14,
    "bank"      : 2,
    "channel"   : 3,
    "bankgroup" : 2,
    "column"    : 5
}

class Command(str, Enum):
    Read = "RD",
    Write = "WR",
    Move =  "RDWR",
    Deliver = "DR",
    Transfer = "TR",
    Prefetch = "PR",
    Compute_Delay = "DL"
    Read_DIMM = "RD_DIMM"

class EmbTableProfiler:

    table_profiles = []
    table_index = []

    @staticmethod
    def set_table_profile(table_id, table_len):
        table_pf = []
        for i in range(table_len):
            table_pf.append(np.zeros(1))
        EmbTableProfiler.table_profiles.append(table_pf)

    @staticmethod
    def record_profile(table_id, vec_ids):
        for vec_id in vec_ids:
            EmbTableProfiler.table_profiles[table_id][vec_id] += 1

def save_profile_result(path, dataset):
    savefile = f'./{path}/{dataset}/profile.pickle'
    profiles = EmbTableProfiler.table_profiles
    with open(savefile, 'wb') as sf:
        pickle.dump(profiles, sf)

def load_profile_result(path, dataset):
    savefile = f'{path}/{dataset}/profile.pickle'
    profiles = None
    print(f"Loading embedding profile from {savefile}")
    if not os.path.exists(savefile):
        print('please run dlrm first!')
        sys.exit()
    else:
        with open(savefile, 'rb') as wf:
            profiles = pickle.load(wf)            

    return profiles

def load_criteo_train_data(path='./savedata/', dataset='kaggle'):
    print('Loading train data for trace generation')
    train_data = None
    train_data_savefile= os.path.join(path, dataset, "train_data.pickle")
    if not os.path.exists(train_data_savefile):
        print("read from kaggle")
        if dataset == 'kaggle':
            train_data = dp.CriteoDataset(
                "kaggle",
                -1,
                0.0,
                "total",
                "train",
                "./input/train.txt",
                "./share_st/kaggle/input/kaggleAdDisplayChallenge_processed.npz",
                False,
                False
            )
        else:
            print("read from terabyte")
            train_data = dp.CriteoDataset(
                "terabyte",
                10000000,
                0.0,
                "total",
                "train",
                "./terabyte_input/day",
                "./terabyte_input/terabyte_processed.npz",
                True,
                False
            )
        with open(train_data_savefile, 'wb') as savefile:
                pickle.dump(train_data, savefile)
        pass
    else:
        with open(train_data_savefile, 'rb') as loadfile:
            train_data = pickle.load(loadfile)

    return train_data

def get_bg_id(addr:int):
    shift_bits = 6
    bg_start_bit = addr_map["column"]
    bg_end_bit = bg_start_bit + addr_map["bankgroup"]
    bg_id = (addr >> (bg_start_bit + shift_bits)) & ((1 << (addr_map["bankgroup"]) - 1))
    
    return bg_id

def write_trace_line(wf, device:str, physical_addr:int, command:str, total_burst:int):
    wf.write(f"{device} {command} {physical_addr} {int(total_burst)} \n")

def data_move(wf, device:str, addr_mapper:ProactivePIMTranslation, addr:int, cmp_addr:int, pim_level:str, burst:int):
    write_trace_line(wf, device, addr, Command.Read, burst)
    transfer_addr, same_node = addr_mapper.map_to_same_node(pim_level, cmp_addr, addr, randomize_row=True)
    if not same_node:
        write_trace_line(wf, device, transfer_addr, Command.Deliver, burst)

def write_trace_file(
        embedding_profiles=None,
        train_data=None,
        dataset='kaggle',
        total_trace=10,
        batch_size=4,
        collisions=4,
        tt_rank=16, 
        vec_size=64,
        cpu_baseline=False,
        cache=None,
        table_prefetch=True,
        all_prefetch=False,
        using_subtable_mapping=True,
        addr_mappers=[],
        pim_level="bankgroup",
        using_skinny_gemm=False
    ):

    print("Generating traces for DRAMsim3")

    if len(addr_map) != 6:
        print("please provide correct address mapping!")
        sys.exit()

    total_batch = total_trace // batch_size
    default_vec_size = 64
    total_burst = vec_size // default_vec_size
    tt_rec_burst = tt_rank * 4 // default_vec_size
    tt_rec_burst_pow_2 = tt_rank * tt_rank * 4 * 4 // default_vec_size
    total_data = len(train_data)
    mlp_arch = 13 * 512 * 256 * 64 * 16 # approximately 6GB
    mlp_bursts = mlp_arch * 4 / 64
    HBM_DIMM2_bw_ratio = 5
    mlp_start_addr = 4 * 1024**3  # 4GB in bytes
    mlp_end_addr = mlp_arch * 4 # 4 + 6GB reaches almost 8GB capacity of DDR4
    HBM_clk_delay = 1 / math.pow(10, 12) # 1ns
    tt_delay = math.ceil((tt_rank*tt_rank + tt_rank) / (0.98 * math.pow(10,12)) * vec_size / HBM_clk_delay)

    multi_hot = Multihot(
                    multi_hot_sizes=[10 for i in range(len(embedding_profiles))],
                    num_embeddings_per_feature=[len(table) for table in embedding_profiles],
                    batch_size=1,
                    collect_freqs_stats=False,
                    dist_type='pareto',
                    dataset=dataset
                )

    for addr_mapper in addr_mappers:
        mapper_name = addr_mapper.mapper_name()
        using_prefetch = all_prefetch or table_prefetch and "ProactivePIM" in mapper_name
        is_QR = True if "QR" in mapper_name else False
        is_TT_Rec = True if "TT" in mapper_name else False
        
        wfile = f'./traces/{dataset}/{mapper_name}_{dataset}_{vec_size}B_'        
        if cpu_baseline:
            wfile = wfile + 'baseline.txt'
        elif using_subtable_mapping:
            if table_prefetch:
                wfile = wfile + 'table_prefetch.txt'
            elif all_prefetch:
                wfile = wfile + 'all_prefetch.txt'
            else:
                wfile = wfile + 'submap.txt'
        else:
            wfile = wfile + 'normal_pim.txt'
        print(f"write file name : {wfile}")


        total_bg = int(math.pow(2,addr_map['bankgroup']) * math.pow(2, addr_map['rank']) * math.pow(2,addr_map['channel']))

        with open(wfile, 'w') as wf:
            total_data_move = 0
            cache_hit = 0
            load_per_bg = [0 for i in range(total_bg)]

            for i in range(len(train_data)//batch_size):
                batch_data = [feat for _, feat, _ in train_data[i*batch_size:(i+1)*batch_size]]
                batch_data = np.array(batch_data, dtype=np.int64)
                batch_data = np.transpose(batch_data)
                multi_hot_indices = multi_hot.make_new_batch(lS_i=batch_data, batch_size=batch_size)
                multi_hot_indices = np.transpose(multi_hot_indices)
                # print(multi_hot_indices.shape)

                if i % 2 == 0:
                    print(f"{i}/{total_batch} trace processed")
                if i > total_batch:
                    break

                # prefetch only once if all_prefetch flag is set
                if all_prefetch and i == 0:
                    for table in range(len(embedding_profiles)):
                        prefetch_addrs = addr_mapper.get_prefetch_physical_address(table)
                        burst = tt_rec_burst if is_TT_Rec else total_burst
                        write_trace_line(wf, device, prefetch_addrs[i], Command.Prefetch, burst)

                for table, batch_embs in enumerate(multi_hot_indices):     
                    mlp_load_count = 0
                    # write transfer/prefetch command for each table
                    # inital execution has no result to transfer // batch_size * 4 => four bank groups
                    transfers = 4*batch_size if not (i == 0 and table == 0) else 0 
                    if using_prefetch:
                        prefetch_addrs = addr_mapper.get_prefetch_physical_address(table)
                        overhead = max(transfers, len(prefetch_addrs))
                        for i in range(overhead):
                            device = "HBM"
                            if i < transfers:
                                write_trace_line(wf, device, 0, Command.Transfer, total_burst)
                            if i < len(prefetch_addrs):
                                burst = tt_rec_burst if is_TT_Rec else total_burst
                                write_trace_line(wf, device, prefetch_addrs[i], Command.Prefetch, burst)
                    elif not cpu_baseline:
                        for i in range(transfers):
                            write_trace_line(wf, device, 0, Command.Transfer, total_burst)

                    total_emb_bursts = 0
                    for l, multi_embs in enumerate(batch_embs):
                        for emb in multi_embs:
                            emb = int(emb.item())
                            if is_QR:
                                (q_addr, r_addr), (q_cmd, r_cmd) = addr_mapper.physical_translation(table, emb)
                                device = "HBM"

                                if cpu_baseline:
                                    for m in range(total_burst):
                                        tmp_addr = q_addr + 64*m
                                        q_hit = cache.access(tmp_addr, 'emb')    
                                        if not q_hit:
                                            write_trace_line(wf, device, tmp_addr, Command.Read, 1)    
                                            total_emb_bursts += 1                   
                                    for m in range(total_burst):
                                        tmp_addr = r_addr + 64*m
                                        r_hit = cache.access(tmp_addr, 'emb')
                                        if not r_hit:
                                            write_trace_line(wf, device, tmp_addr, Command.Read, 1)
                                            total_emb_bursts += 1

                                    # concurrent access to MLP if cpu_baseline flag is set (to mimic cache conflict behavior)
                                    if mlp_load_count < mlp_bursts:
                                        total_mlp_load = total_emb_bursts // HBM_DIMM2_bw_ratio
                                        leftovers = total_emb_bursts % HBM_DIMM2_bw_ratio
                                        if total_mlp_load > 0:
                                            for k in range(total_mlp_load):
                                                addr = random.randint(mlp_start_addr, mlp_end_addr)
                                                cache.access(addr, 'mlp')
                                                mlp_load_count += 1
                                            total_emb_bursts = leftovers

                                else:
                                    if q_cmd == Command.Read_DIMM:
                                        write_trace_line(wf, "DIMM", q, Command.Read, total_burst)
                                    else:
                                        write_trace_line(wf, device, q_addr, q_cmd, total_burst)     
                                        load_per_bg[get_bg_id(q_addr)] += 1                           
                                    if not using_prefetch:
                                        if using_subtable_mapping:
                                            write_trace_line(wf, device, r_addr, r_cmd, total_burst)
                                        else:
                                            if r_cmd == Command.Move:
                                                data_move(wf, device, addr_mapper, r_addr, q_addr, pim_level, total_burst)
                                                total_data_move += 1
                                            else:
                                                write_trace_line(wf, device, r_addr, r_cmd, total_burst)


                            elif is_TT_Rec:
                                total_access = addr_mapper.physical_translation(table, emb)
                                device = "HBM"

                                if cpu_baseline:
                                    cache.flush()
                                
                                for (a, b, c), (first_cmd, second_cmd, third_cmd) in total_access:
                                    if cpu_baseline:
                                        for m in range(tt_rec_burst):
                                            a_addr = a + 64*m
                                            a_hit = cache.access(a_addr, 'emb')
                                            if not a_hit:
                                                write_trace_line(wf, device, a_addr, Command.Read, 1)   
                                                total_emb_bursts += tt_rec_burst

                                        for m in range(tt_rec_burst):
                                            c_addr = c + 64*m
                                            c_hit = cache.access(c_addr, 'emb')
                                            if not c_hit:
                                                write_trace_line(wf, device, c_addr, Command.Read, 1)    
                                                total_emb_bursts += tt_rec_burst
                                         
                                        # sample 3 vectors out of total tt_rank vectors to avoid enormous trace file
                                        # write_idx = random.sample(range(tt_rec_burst_pow_2), tt_rec_burst * 3)
                                        for m in range(tt_rec_burst_pow_2):
                                            b_addr = b + 64*m
                                            b_hit = cache.access(b_addr, 'emb')
                                            if not b_hit:
                                                total_emb_bursts += tt_rec_burst
                                                # if m in write_idx:
                                                write_trace_line(wf, device, b_addr, Command.Read, 1)    
                                        
                                        # write_trace_line(wf, device, 0, Command.Compute_Delay, tt_delay)    
                                        
                                        # concurrent access to MLP if cpu_baseline flag is set (to mimic cache conflict behavior)
                                        if mlp_load_count < mlp_bursts:
                                            total_mlp_load = total_emb_bursts // HBM_DIMM2_bw_ratio
                                            leftovers = total_emb_bursts % HBM_DIMM2_bw_ratio
                                            if total_mlp_load > 0:
                                                for k in range(total_mlp_load):
                                                    mlp_addr = mlp_start_addr + mlp_load_count
                                                    cache.access(mlp_addr, 'mlp')
                                                    mlp_load_count += 1
                                                total_emb_bursts = leftovers
                                                
                                    else:
                                        if using_skinny_gemm:
                                            if not (a == -1):
                                                write_trace_line(wf, device, a, first_cmd, tt_rec_burst)
                                            if not (b == -1):
                                                write_trace_line(wf, device, b, Command.Read, tt_rec_burst)
                                            if not (c == -1):
                                                write_trace_line(wf, device, c, third_cmd, tt_rec_burst)
                                        else:
                                            if using_subtable_mapping:
                                                # using intermediate result of a and b
                                                if not (a == -1):
                                                    write_trace_line(wf, device, a, first_cmd, tt_rec_burst)
                                                write_trace_line(wf, device, c, third_cmd, tt_rec_burst)
                                            else:
                                                if first_cmd == Command.Move:
                                                    total_data_move += 1
                                                    data_move(wf, device, addr_mapper, a, b, pim_level, tt_rec_burst)
                                                else:
                                                    write_trace_line(wf, device, a, Command.Read, tt_rec_burst)

                                                if third_cmd == Command.Move:
                                                    total_data_move += 1
                                                    data_move(wf, device, addr_mapper, c, b, pim_level, tt_rec_burst)
                                                else:
                                                    write_trace_line(wf, device, c, Command.Read, tt_rec_burst)

                                                write_trace_line(wf, device, b, Command.Read, tt_rec_burst_pow_2)


                                            # # if not (b == -1):
                                            # else:
                                            #     if second_cmd == Command.Read_DIMM:
                                            #         write_trace_line(wf, "DIMM", b, Command.Read, tt_rec_burst)

                            else: 
                                total_burst = vec_size // default_vec_size
                                addr = addr_mapper.physical_translation(table, emb)
                                device = "HBM"
                                write_trace_line(wf, device, addr, Command.Read, total_burst)

                    wf.write('\n')
            
            print("total_data_move : ", total_data_move)
            print("cache hit rate : ", cache.overall_hit_rate())
            print("emb hit rate : ", cache.category_hit_rate('emb'))
            # if not cpu_baseline:
            #     print("bg loads : ", np.array(load_per_bg)/np.min(load_per_bg))            

def addrmap_generator(
        mapper_name='ProactivePIM',
        embedding_profiles=None,
        qr=True,
        tt_rec=False,
        vec_size=128, 
        collision=4, 
        tt_rank=16, 
        using_prefetch=False, 
        using_mapping=False, 
        using_skinny_gemm=False,
        pim_level='bankgroup', 
        cmp_ch_only=False,
    ):
    
    addr_mappers = []
    
    if qr:
        addr_mappers.append(
                ProactivePIMTranslation(
                    embedding_profiles=embedding_profiles, 
                    vec_size=vec_size, 
                    HBM_size_gb=4, 
                    is_QR=True, 
                    collisions=collision, 
                    using_prefetch=using_prefetch,
                    using_subtable_mapping=using_mapping,
                    addr_map=addr_map,
                    pim_level=pim_level,
                    cmp_ch_only=cmp_ch_only,
                    mapper_name=mapper_name+"_QR"
                )
        )

    if tt_rec:
        addr_mappers.append(
                ProactivePIMTranslation(
                    embedding_profiles=embedding_profiles, 
                    vec_size=vec_size, 
                    HBM_size_gb=4, 
                    is_TT_Rec=True, 
                    tt_rank=tt_rank, 
                    using_prefetch=using_prefetch,
                    using_subtable_mapping=using_mapping,
                    using_skinny_gemm=using_skinny_gemm,
                    addr_map=addr_map,
                    pim_level=pim_level,
                    cmp_ch_only=cmp_ch_only,
                    mapper_name=mapper_name+"_TT"
                )
        )

    return addr_mappers

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="kaggle", help="dataset : kaggle or terabyte")
    parser.add_argument('--data_path', type=str, default='./savedata', help="path for train data and embedding profile")
    parser.add_argument('--qr', type=bool, default=False, help="using qr") 
    parser.add_argument('--tt', type=bool, default=False, help="using tt")
    parser.add_argument('--vec_sizes', type=int, nargs="*", default=[128], help="vector size in list format")
    parser.add_argument('--batch', type=int, default=4, help="batch size")
    parser.add_argument('--collision', type=int, default=4, help="qr collision") 
    parser.add_argument('--tt_rank', type=int, default=16, help="tt rank")
    parser.add_argument('--pim_level', type=str, default='bankgroup', help="pim level : rank, bankgroup, bank")
    parser.add_argument('--baseline', type=bool, default=False, help="baseline trace")
    parser.add_argument('--submap', type=bool, default=False, help="subtable mapping trace")
    parser.add_argument('--allprefetch', type=bool, default=False, help="all prfetch trace")
    parser.add_argument('--tableprefetch', type=bool, default=False, help="table prefetch trace")
    parser.add_argument('--data_move_mode', type=str, default="bankgroup", help="bankgroup move or channel move")

    args = parser.parse_args()    

    dataset = args.dataset
    data_path = args.data_path
    vec_sizes = args.vec_sizes
    batch_size = args.batch
    collision = args.collision
    using_tt = args.tt
    using_qr = args.qr
    tt_rank = args.tt_rank
    dataset = args.dataset
    pim_level = args.pim_level
    cpu_baseline = args.baseline
    using_subtable_mapping = args.submap
    table_prefetch = args.tableprefetch
    all_prefetch = args.allprefetch
    
    data_move_channel_only = False
    if args.data_move_mode == "channel":
        data_move_channel_only = True

    scale_factor = 1
    if using_tt:
        scale_factor = 0.077/4 # cannot get a dataset that is near 4GB after tt-rec compression (128B). Scale cache size instead.
    elif using_qr:
        scale_factor = 1/4
    MB_size = 2**20
    CACHE_SIZE = int(32*MB_size*scale_factor)  # in bytes
    BLOCK_SIZE = 64    # in bytes
    ASSOCIATIVITY = 4
        
    cache = CacheSimulator.Cache(CACHE_SIZE, BLOCK_SIZE, ASSOCIATIVITY)

    data_path = './savedata'
    train_data = load_criteo_train_data(data_path, dataset=dataset)
    embedding_profiles = load_profile_result(data_path, dataset=dataset)

    for vec_size in vec_sizes:
        using_prefetch = table_prefetch or all_prefetch
        print("Generating address mappers")
        ProactivePIM_maps = addrmap_generator(
                                    mapper_name="ProactivePIM",
                                    embedding_profiles=embedding_profiles, 
                                    vec_size=vec_size, 
                                    qr=using_qr,
                                    tt_rec=using_tt,
                                    collision=collision, 
                                    tt_rank=tt_rank, 
                                    using_prefetch=using_prefetch, 
                                    using_mapping=using_subtable_mapping, 
                                    using_skinny_gemm=((not cpu_baseline) and using_subtable_mapping),
                                    pim_level="bankgroup",
                                    cmp_ch_only=data_move_channel_only
                                )
        # ProactivePIM_maps = addrmap_generator(embedding_profiles, vec_size, collision, tt_rank, using_prefetch, using_subtable_mapping, "bankgroup", "ProactivePIM_All_Prefetch")
        # RecNMP_maps = addrmap_generator(embedding_profiles, vec_size, collision, tt_rank, False, using_subtable_mapping, "rank", "RecNMP")
        # SPACE_maps = addrmap_generator(embedding_profiles, vec_size, collision, tt_rank, False, using_subtable_mapping, "rank", "SPACE")
        # addr_mappers = list(chain(ProactivePIM_maps, RecNMP_maps, SPACE_maps))
        addr_mappers = ProactivePIM_maps
        
        write_trace_file(
                embedding_profiles=embedding_profiles,
                train_data=train_data,
                dataset=dataset,
                total_trace=100,
                collisions=collision,
                tt_rank=tt_rank,
                vec_size=vec_size,
                batch_size=batch_size,
                table_prefetch=table_prefetch,
                all_prefetch=all_prefetch,
                using_subtable_mapping=using_subtable_mapping,
                addr_mappers=addr_mappers,
                cpu_baseline=cpu_baseline,
                cache=cache,
                pim_level=pim_level,
                using_skinny_gemm=((not cpu_baseline) and using_subtable_mapping)
            )