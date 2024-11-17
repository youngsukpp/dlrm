import math
import numpy as np
import random
import pickle
import sys
import os
from abc import *

class WeightSharingTranslator():
    def __init__(
        self,
        embedding_profiles,
        collision,
        tt_rank,
        vec_size
    ):

        self.embedding_profiles = embedding_profiles
        self.collision = collision
        self.tt_rank = tt_rank
        self.vec_size = vec_size

        self.Q_entries_per_table, self.R_entries_per_table = self.preprocess_QR()
        self.core_info = self.preprocess_TT_Rec()

    def preprocess_QR(self):
        Q_entries_per_table = [int(self.embedding_profiles[i].shape[0]/self.collision) for i in range(len(self.embedding_profiles))]
        R_entries_per_table = [self.collision for i in range(len(self.embedding_profiles))]
        return Q_entries_per_table, R_entries_per_table

    def preprocess_TT_Rec(self):
        core_entries_per_table = []
        for i in range(len(self.embedding_profiles)):
            entry = self.find_number_and_combination_entries(len(self.embedding_profiles[i]))
            core_entries_per_table.append(entry)

        # for j in range(len(self.vec_size)):
        total_elements = self.vec_size // 4
        vec_dims = self.find_number_and_combination_for_vec(total_elements)      

        # core_info = {}
        # for k in range(len(self.vec_size)):
        core_info = (vec_dims, core_entries_per_table)

        return core_info

    def get_QR_size(self, table_idx, vec_size, is_Q):
        if is_Q:
            return self.Q_entries_per_table[table_idx] * vec_size
        else:
            return self.R_entries_per_table[table_idx] * vec_size

    def get_TT_Rec_size(self, table_idx, vec_size, is_first, is_second, is_third):
        core_dims, cores_entries_per_table = self.core_info
        cores_entries = cores_entries_per_table[table_idx]

        if is_first:
            core_dim = core_dims[0] * cores_entries * self.tt_rank * 4
        elif is_second:
            core_dim = core_dims[1] * cores_entries * self.tt_rank * self.tt_rank * 4
        elif is_third:
            core_dim = core_dims[2] * cores_entries * self.tt_rank * 4

        return core_dim

    def get_QR_entry(self, vec_size, table_idx, vec_idx):
        return vec_idx // self.collision, vec_idx % self.collision

    def get_TT_Rec_entry(self, vec_size, table_idx, vec_idx):
        elements = vec_size // 4
        core_dims, core_entries_per_table = self.core_info
        cores_entries = core_entries_per_table[table_idx]
        access = []
        for i in range(elements):
            a = vec_idx % cores_entries * core_dims[0] + i % core_dims[0]
            b = ((vec_idx % math.pow(cores_entries,2)) // cores_entries) * core_dims[1] + ((i % (core_dims[0] * core_dims[1])) // core_dims[0])
            c = vec_idx // math.pow(cores_entries, 2) * core_dims[2] + i // (core_dims[0] * core_dims[1])
            access.append((a,b,c))

        return access

    def find_number_and_combination_entries(self, N):
        x = 1
        while (x + 1) ** 3 <= N:
            x += 1

        largest_x_cubed = x ** 3

        y = x
        while (y ** 3) < N:
            y += 1

        new_combination_value = y ** 3
        
        return y

    def find_number_and_combination_for_vec(self, N):
        x = 2
        while (x + 2) ** 3 <= N:
            x += 2

        largest_x_cubed = x ** 3

        y1, y2, y3 = x, x, x
        i = 0
        while y1 * y2 * y3 < N:
            if i==0:
                y1 += 2
            elif i==1:
                y2 += 2
            else:
                y3 += 2

            i = (i+1)%3

        new_combination_product = y1 * y2 * y3
        
        return (y1, y2, y3)

class ProactivePIMTranslation():
    def __init__(
        self, 
        embedding_profiles, 
        HBM_size_gb=4, 
        HBM_BW=256,
        vec_size=64,
        is_QR=False,
        collisions=8,
        is_TT_Rec=False,
        using_prefetch=False,
        using_subtable_mapping=False,
        using_gemv_dist=True,
        pim_level="bankgroup",
        tt_rank=16,
        addr_map={},
        mapper_name="ProactivePIM"
        # DIMM_size_gb=16, 
        # DIMM_BW=25.6,
    ):
   
        self.embedding_profiles = embedding_profiles
        self.is_QR = is_QR
        self.is_TT_Rec = is_TT_Rec
        self.vec_size = vec_size
        self.collisions = collisions
        self.tt_rank = tt_rank
        self.pim_level = pim_level
        self.using_gemv_dist = using_gemv_dist

        self.addr_map = addr_map
        self.using_prefetch = using_prefetch
        self.using_subtable_mapping = using_subtable_mapping

        self.nodes = HBM_size_gb // 0.5
        self.page_offset = math.pow(2, 12)
        # self.hot_vec_loc = self.profile_hot_vec_location()
        self.mapper_name_ = mapper_name

        # DIMM size and ppns
        self.GB_size = math.pow(2, 30)
        self.HBM_Size = HBM_size_gb * self.GB_size
        # self.DIMM_Size = DIMM_size_gb * self.GB_size
        self.HBM_max_page_number = int(self.HBM_Size // self.page_offset)
        # self.DIMM_max_page_number = int(self.DIMM_Size // self.page_offset)

        # preprocess for logical2physical translation
        self.ws_translator = WeightSharingTranslator(embedding_profiles, collisions, tt_rank, vec_size=self.vec_size)
        self.page_translation_HBM = self.preprocess()

    def mapper_name(self):
        return self.mapper_name_

    def get_rank(self):
        return self.tt_rank

    def preprocess(self):
        # Reserve space for subtable duplication
        self.r_size = 0
        self.first_size = 0
        self.third_size = 0
        prefetch_info_filename = f"prefetch_info_{self.vec_size}"

        if self.is_QR:
            self.r_size_per_table = self.vec_size * self.collisions
            self.reserved_page = int((self.r_size_per_table*len(self.embedding_profiles)) // self.page_offset) * self.nodes
            prefetch_info_filename += "_QR"
            with open(prefetch_info_filename, 'w') as wf:
                for i in range(len(self.embedding_profiles)):
                    start = self.r_size_per_table * i
                    end = self.r_size_per_table * (i+1)
                    wf.write(f"{start} {end}")

        elif self.is_TT_Rec:
            self.first_size_per_table = np.array([self.ws_translator.get_TT_Rec_size(i, self.vec_size, True, False, False) for i in range(len(self.embedding_profiles))])
            self.third_size_per_table = np.array([self.ws_translator.get_TT_Rec_size(i, self.vec_size, False, False, True) for i in range(len(self.embedding_profiles))])
            self.reserved_page = int((np.sum(self.first_size_per_table)+np.sum(self.third_size_per_table))//self.page_offset) * self.nodes

            prefetch_info_filename += "_TT_Rec"
            with open(prefetch_info_filename, 'w') as wf:
                for i in range(len(self.embedding_profiles)):
                    start = np.sum(self.first_size_per_table[:i] + self.third_size_per_table[:i])
                    end = start + self.first_size_per_table[i] + self.third_size_per_table[i]
                    wf.write(f"{start} {end}")
        else:
            self.reserved_page = 0

        # preprocess for logical2physical translation
        self.table_addr_HBM = self.logical_translation(self.embedding_profiles, self.vec_size, self.collisions)
        page_translation_HBM = [i for i in range(self.HBM_max_page_number)]
        # self.page_translation_DIMM = [i for i in range(self.DIMM_max_page_number)]
        random.shuffle(page_translation_HBM)
        # random.shuffle(self.page_translation_DIMM)

        return page_translation_HBM

    def logical_translation(self, embedding_profiles, vec_size, collisions):
        # logical address of Q table and 2nd core
        if self.is_QR:
            space_per_table_HBM = [(self.ws_translator.get_QR_size(i, vec_size, True)) for i in range(len(embedding_profiles))]
        elif self.is_TT_Rec:
            if self.using_gemv_dist:
                space_per_table_HBM = [(self.ws_translator.get_TT_Rec_size(i, vec_size, False, True, False))/self.tt_rank for i in range(len(embedding_profiles))]
            else:
                space_per_table_HBM = [(self.ws_translator.get_TT_Rec_size(i, vec_size, False, True, False)) for i in range(len(embedding_profiles))]
        else:        
            space_per_table_HBM = [(len(embedding_profiles[i]) * vec_size) for i in range(len(embedding_profiles))]

        table_addr_HBM = []
        HBM_accumulation = 0 
        HBM_accumulation += self.reserved_page
        if self.is_TT_Rec:
            if self.using_gemv_dist:
                empty_space = self.HBM_Size - np.sum(space_per_table_HBM) * self.tt_rank
                for i in range(len(space_per_table_HBM) * self.tt_rank):
                    table_addr_HBM.append(HBM_accumulation)
                    HBM_accumulation += space_per_table_HBM[i%len(space_per_table_HBM)] + empty_space/(len(space_per_table_HBM) * self.tt_rank)
            else:
                empty_space = self.HBM_Size - np.sum(space_per_table_HBM)
                for i in range(len(space_per_table_HBM)):
                    table_addr_HBM.append(HBM_accumulation)
                    HBM_accumulation += space_per_table_HBM[i] + empty_space/len(space_per_table_HBM)
        elif self.is_QR:
            empty_space = self.HBM_Size - np.sum(space_per_table_HBM)
            for i in range(len(space_per_table_HBM)):
                table_addr_HBM.append(HBM_accumulation)
                # print(space_per_table_HBM)
                HBM_accumulation += space_per_table_HBM[i] + empty_space/len(space_per_table_HBM)
        else:
            for i in range(len(space_per_table_HBM)):
                table_addr_HBM.append(HBM_accumulation)
                HBM_accumulation += space_per_table_HBM[i]


        print("HBM occupied: ", HBM_accumulation/1024/1024/1024, "GB")

        return table_addr_HBM

    def get_prefetch_physical_address(self, table_idx):
        total_addr = []
        if self.is_QR:
            for i in range(self.collisions):
                (_, r_physical_addr), _ = self.physical_translation(table_idx, i)
                total_addr.append(r_physical_addr)
        elif self.is_TT_Rec:
            first_core_size = self.first_size_per_table[table_idx]
            total_entries = int(first_core_size / self.tt_rank / 4)
            table_start_addr = np.sum(self.first_size_per_table[:table_idx])
            for i in range(total_entries):
                addr = table_start_addr + i * self.tt_rank * 4
                total_addr.append(addr)

        return total_addr

    def int_to_binary(self, addr, bits):
        """ Convert integer to binary string with given bit width """
        return bin(addr)[2:].zfill(bits)

    def binary_to_int(self, binary_str):
        """ Convert binary string to integer """
        return int(binary_str, 2)

    def merge_address(self, parsed_addr):
        """ Merge the parsed address back to an integer """
        binary_addr = ''.join(parsed_addr[field] for field in self.addr_map if self.addr_map[field] > 0)
        return self.binary_to_int(binary_addr)

    def parse_address(self, addr):
        """ Parse the address according to addr_map """
        binary_addr = self.int_to_binary(int(addr), sum(self.addr_map.values()))
        parsed_addr = {}
        index = 0
        for field, bits in reversed(self.addr_map.items()):
            if bits > 0:
                parsed_addr[field] = binary_addr[index:index+bits]
                index += bits
        return parsed_addr

    def map_to_same_node(self, pim_level, target_addr, to_addr, randomize_row=False):
        # Parse the addresses
        target_parsed = self.parse_address(int(target_addr))
        to_parsed = self.parse_address(int(to_addr))

        already_same_chbg = (target_parsed["bankgroup"] == to_parsed["bankgroup"]) and (target_parsed["channel"] == to_parsed["channel"])

        # Modify target's bankgroup and channel to match to_addr's
        target_parsed["bankgroup"] = to_parsed["bankgroup"]
        target_parsed["channel"] = to_parsed["channel"]

        # Optionally randomize the row
        if randomize_row:
            max_row_value = (1 << self.addr_map["row"]) - 1  # Calculate the max value based on the bit width
            random_row = random.randint(0, max_row_value)
            target_parsed["row"] = self.int_to_binary(random_row, self.addr_map["row"])

        # Merge back to integer
        new_target_addr = self.merge_address(target_parsed)
        return new_target_addr, already_same_chbg

    def compare_channel_and_bankgroup(self, addr1, addr2):
        # Parse the addresses
        parsed_addr1 = self.parse_address(addr1)
        parsed_addr2 = self.parse_address(addr2)

        # Compare channel and bankgroup
        same_channel = parsed_addr1["channel"] == parsed_addr2["channel"]
        same_bankgroup = parsed_addr1["bankgroup"] == parsed_addr2["bankgroup"]

        return not (same_channel and same_bankgroup)

    def physical_translation(self, table_idx, vec_idx):    
        ## R table and 1st, 3rd core is located at the front of the table
        # generate ppn
        if self.is_QR:
            table_logical_addr = self.table_addr_HBM[table_idx]
            q_idx, r_idx = self.ws_translator.get_QR_entry(self.vec_size, table_idx, vec_idx)
            q_vec_logical_addr = table_logical_addr + q_idx * self.vec_size
            r_vec_logical_addr = table_idx * self.collisions * self.vec_size + r_idx * self.vec_size

            q_vpn = int(q_vec_logical_addr // self.page_offset)
            q_ppn = self.page_translation_HBM[q_vpn]
            q_po_loc = q_vec_logical_addr % self.page_offset

            r_vpn = int(r_vec_logical_addr // self.page_offset)
            r_ppn = r_vpn
            r_po_loc = r_vec_logical_addr % self.page_offset

            q_physical_addr = int(q_ppn*self.page_offset + q_po_loc)
            r_physical_addr = int(r_ppn*self.page_offset + r_po_loc)

            r_command = "RD"
            if self.using_subtable_mapping:
                r_physical_addr, _ = self.map_to_same_node(self.pim_level, q_physical_addr, r_physical_addr)
            else:
                need_transfer_to_other_node = self.compare_channel_and_bankgroup(q_physical_addr, r_physical_addr)
                if need_transfer_to_other_node:
                    r_command = "RDWR"
            
            if self.using_prefetch:
                r_command = "RDD" # read every duplicated r addr from every node

            return (q_physical_addr, r_physical_addr), ("RD", r_command)
        
        elif self.is_TT_Rec:
            access = self.ws_translator.get_TT_Rec_entry(self.vec_size, table_idx, vec_idx)
            total_physical_addr = []
            for a, b, c in access:
                if self.using_gemv_dist:
                    for k in range(4):
                        rank = k*4
                        first_c_logical_addr = np.sum(self.first_size_per_table[:table_idx]) + a * self.tt_rank * 4
                        third_c_logical_addr = np.sum(self.third_size_per_table[:table_idx]) + c * self.tt_rank * 4
                        table_logical_addr = self.table_addr_HBM[table_idx + rank*len(self.embedding_profiles)]
                        second_c_logical_addr = rank * table_logical_addr + b * self.tt_rank * 4
                        # distributing second_c_logical_addr across bankgroup
                        # second_c_logical_addr = table_logical_addr + b * self.tt_rank * self.tt_rank * 4

                        # using direct mapping
                        first_c_physical_addr, second_c_physical_addr, third_c_physical_addr = int(first_c_logical_addr), int(second_c_logical_addr), int(third_c_logical_addr)

                        first_c_command = "RD"
                        third_c_command = "RD"
                        if self.using_subtable_mapping:
                            first_c_logical_addr, _ = self.map_to_same_node(self.pim_level, second_c_physical_addr, first_c_physical_addr)
                            third_c_logical_addr, _ = self.map_to_same_node(self.pim_level, second_c_physical_addr, third_c_physical_addr)
                        else:
                            need_transfer_to_other_node_1st = self.compare_channel_and_bankgroup(second_c_physical_addr, first_c_physical_addr)
                            need_transfer_to_other_node_3rd = self.compare_channel_and_bankgroup(second_c_physical_addr, third_c_physical_addr)
                            if need_transfer_to_other_node_1st:
                                first_c_command = "RDWR"
                            if need_transfer_to_other_node_3rd:
                                third_c_command = "RDWR"

                        if self.using_prefetch:
                            first_c_command = "RDD"

                        total_physical_addr.append(((first_c_physical_addr, second_c_physical_addr, third_c_physical_addr), (first_c_command, "RD", third_c_command)))
                else:
                    table_logical_addr = self.table_addr_HBM[table_idx]
                    first_c_logical_addr = np.sum(self.first_size_per_table[:table_idx]) + a * self.tt_rank * 4
                    third_c_logical_addr = np.sum(self.third_size_per_table[:table_idx]) + c * self.tt_rank * 4
                    second_c_logical_addr = table_logical_addr + b * self.tt_rank * self.tt_rank * 4

                    first_c_command = "RD"
                    third_c_command = "RD"
                    if self.using_subtable_mapping:
                        first_c_logical_addr, _ = self.map_to_same_node(self.pim_level, int(second_c_logical_addr), first_c_logical_addr)
                        third_c_logical_addr, _ = self.map_to_same_node(self.pim_level, int(second_c_logical_addr), third_c_logical_addr)
                    else:
                        need_transfer_to_other_node_1st = self.compare_channel_and_bankgroup(second_c_logical_addr, first_c_logical_addr)
                        need_transfer_to_other_node_3rd = self.compare_channel_and_bankgroup(second_c_logical_addr, third_c_logical_addr)
                        if need_transfer_to_other_node_1st:
                            first_c_command = "RDWR"
                        if need_transfer_to_other_node_3rd:
                            third_c_command = "RDWR"

                    if self.using_prefetch:
                        first_c_command = "RDD"

                    first_c_vpn, second_c_vpn, third_c_vpn = int(first_c_logical_addr // self.page_offset), int(second_c_logical_addr // self.page_offset), int(third_c_logical_addr // self.page_offset)
                    first_c_ppn, second_c_ppn, third_c_ppn = first_c_vpn, second_c_vpn, third_c_vpn
                    first_c_po_loc, second_c_po_loc, third_c_po_loc = first_c_logical_addr % self.page_offset, second_c_logical_addr % self.page_offset, third_c_logical_addr % self.page_offset                
                    first_c_physical_addr, second_c_physical_addr, third_c_physical_addr = int(first_c_ppn*self.page_offset + first_c_po_loc), int(second_c_ppn*self.page_offset + second_c_ppn), int(third_c_ppn*self.page_offset + third_c_po_loc)
                    total_physical_addr.append(((first_c_physical_addr, second_c_physical_addr, third_c_physical_addr), (first_c_command, "RD", third_c_command)))

            return total_physical_addr

        else:
            table_logical_addr = self.table_addr_HBM[table_idx]
            logical_addr = table_logical_addr + vec_idx * self.vec_size

            vpn = int(logical_addr // self.page_offset)
            ppn = self.page_translation_HBM[vpn]
            po_loc = logical_addr % self.page_offset

            physical_addr = int(ppn*self.page_offset + po_loc)

            return physical_addr

# Heterogeneous memory system

# def logical_translation(self, hot_vec_loc, embedding_profiles, vec_size, collisions):        
#     if self.is_QR:
#         space_per_table_HBM = [0 + vec_size * (self.reserved_page+len(hot_vec_loc[i])) for i in range(len(hot_vec_loc))]
#         space_per_table_DIMM = [0 + vec_size * (prof_per_table.shape[0]-len(hot_vec_loc[i])) for i, prof_per_table in enumerate(embedding_profiles)]
#     elif self.is_TT_Rec:
#         space_per_table_HBM = [0 + vec_size * (self.reserved_page+collisions+len(hot_vec_loc[i])) for i in range(len(hot_vec_loc))]
#         space_per_table_DIMM = [0 + vec_size * (prof_per_table.shape[0]-len(hot_vec_loc[i])) for i, prof_per_table in enumerate(embedding_profiles)]       
    # table_addr_HBM = []
    # HBM_accumulation = 0 

    # HBM_accumulation += self.reserved_page
    # for i in range(len(space_per_table_HBM)):
    #     table_addr_HBM.append(HBM_accumulation)
    #     HBM_accumulation += space_per_table_HBM[i]
    
    # print("HBM occupied: ", HBM_accumulation/1024/1024/1024, "GB")

    # table_addr_DIMM = []
    # DIMM_accumulation = 0
    # for i in range(len(space_per_table_DIMM)):
    #     table_addr_DIMM.append(DIMM_accumulation)
    #     DIMM_accumulation += space_per_table_DIMM[i]

    # return table_addr_HBM , table_addr_DIMM