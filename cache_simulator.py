import numpy as np
import my_profiler as prof
from address_translation import BasicAddressTranslation
import os 
import random

class CacheBlock:
    def __init__(self, tag):
        self.tag = tag
        self.lru_counter = 0

class CacheSet:
    def __init__(self, associativity):
        self.blocks = [None] * associativity

    def access_block(self, tag):
        for block in self.blocks:
            if block and block.tag == tag:
                block.lru_counter = 0
                return True

        self.load_block(tag)
        return False

    def load_block(self, tag):
        lru_block = max(self.blocks, key=lambda b: b.lru_counter if b else -1)
        if lru_block:
            lru_block.tag = tag
            lru_block.lru_counter = 0
        else:
            for i in range(len(self.blocks)):
                if self.blocks[i] is None:
                    self.blocks[i] = CacheBlock(tag)
                    return

        for block in self.blocks:
            if block:
                block.lru_counter += 1


class Cache: 
    def __init__(self, size, block_size, associativity):
        self.num_sets = size // (block_size * associativity)
        self.associativity = associativity
        self.sets = [CacheSet(associativity) for _ in range(self.num_sets)]
        self.block_size = block_size
        self.associativity = associativity
        self.hits = {'emb': 0, 'mlp': 0}
        self.misses = {'emb': 0, 'mlp': 0}
        
    def flush(self):
        self.sets = [CacheSet(self.associativity) for _ in range(self.num_sets)]        

    def access(self, address, category):
        tag = address // (self.block_size * self.num_sets)
        set_index = (address // self.block_size) % self.num_sets

        if self.sets[set_index].access_block(tag):
            self.hits[category] += 1
            return True
        else:
            self.misses[category] += 1
            return False

    def category_hit_rate(self, category):
        total_accesses = self.hits[category] + self.misses[category]
        return self.hits[category] / total_accesses if total_accesses > 0 else 0

    def overall_hit_rate(self):
        total_hits = sum(self.hits.values())
        total_misses = sum(self.misses.values())
        total_accesses = total_hits + total_misses

        return total_hits / total_accesses if total_accesses > 0 else 0

# def CacheSimulator(train_data=None, collision=4, vec_size=64, Cache_MB=1, Block_size=64):
#     MB_size = 2**20
#     CACHE_SIZE = Cache_MB*MB_size  # in bytes
#     BLOCK_SIZE = Block_size    # in bytes
#     ASSOCIATIVITY = 4
#     cache = Cache(CACHE_SIZE, BLOCK_SIZE, ASSOCIATIVITY)
#     total_burst = vec_size // 64

#     embedding_profiles = prof.load_profile_result('./savedata', 'Terabyte', collision)
#     translator = BasicAddressTranslation(embedding_profiles, collisions=collision, vec_size=vec_size)
#     total_data = len(train_data)

#     for i, data in enumerate(train_data):
#         _, feat, _ = data
#         if i > 100000:
#             break

#         for table, emb in enumerate(feat):
#             q_emb = emb // collision
#             r_emb = emb % collision
#             _, q_pa = prof.get_physical_address(
#                         addr_translator=translator,
#                         using_bg_map=False,
#                         table_index=table,
#                         vec_index=q_emb,
#                         is_r_vec=False
#                         )
#             _, r_pa = prof.get_physical_address(
#                         addr_translator=translator,
#                         using_bg_map=False,
#                         table_index=table,
#                         vec_index=r_emb,
#                         is_r_vec=True
#                         )
#             for j in range(total_burst):
#                 cache.access(q_pa + 64*j, 'q')
#                 cache.access(r_pa + 64*j, 'r')

#     q_hitrate = round(cache.category_hit_rate('q'), 3)
#     r_hitrate = round(cache.category_hit_rate('r'), 3)
#     total_hitrate = round(cache.overall_hit_rate(), 3)

#     print(f"\nCache Simulation on {Cache_MB}MB Cache/ {Block_size}B block / Vector {vec_size}B / collision {collision}\n")
#     print(f"Q hitrate : {q_hitrate}") 
#     print(f"R hitrate : {r_hitrate}")
#     print(f"Total hitrate : {total_hitrate}")

# def ProactivePIMCacheSimulator(train_data=None, collision=4, vec_size=64, Cache_MB=1, Block_size=64):
#     MB_size = 2**20
#     CACHE_SIZE = Cache_MB*MB_size  # in bytes
#     BLOCK_SIZE = Block_size    # in bytes
#     ASSOCIATIVITY = 4
#     cache = Cache(CACHE_SIZE, BLOCK_SIZE, ASSOCIATIVITY)
#     total_burst = vec_size // 64

#     embedding_profiles = prof.load_profile_result('./savedata', 'Terabyte', collision)
#     translator = BasicAddressTranslation(embedding_profiles, collisions=collision, vec_size=vec_size)
#     total_data = len(train_data)

#     for i, data in enumerate(train_data):
#         _, feat, _ = data
#         if i > 100000:
#             break

#         for table, emb in enumerate(feat):
#             q_emb = emb // collision
#             r_emb = emb % collision
#             _, q_pa = prof.get_physical_address(
#                         addr_translator=translator,
#                         using_bg_map=False,
#                         table_index=table,
#                         vec_index=q_emb,
#                         is_r_vec=False
#                         )
#             _, r_pa = prof.get_physical_address(
#                         addr_translator=translator,
#                         using_bg_map=False,
#                         table_index=table,
#                         vec_index=r_emb,
#                         is_r_vec=True
#                         )
#             for j in range(total_burst):
#                 cache.access(q_pa + 64*j, 'q')
#                 cache.access(r_pa + 64*j, 'r')

#     q_hitrate = round(cache.category_hit_rate('q'), 3)
#     r_hitrate = round(cache.category_hit_rate('r'), 3)
#     total_hitrate = round(cache.overall_hit_rate(), 3)

#     print(f"\nCache Simulation on {Cache_MB}MB Cache/ {Block_size}B block / Vector {vec_size}B / collision {collision}\n")
#     print(f"Q hitrate : {q_hitrate}") 
#     print(f"R hitrate : {r_hitrate}")
#     print(f"Total hitrate : {total_hitrate}")


# def RunCacheSimulation(called_inside_DLRM=False, train_data=None, collision=8, dataset='kaggle'):
#     if not called_inside_DLRM:
#         if dataset=='kaggle':
#             train_data_savefile = './savedata/train_data.pickle'
#             train_data = prof.load_train_data(train_data_savefile)
#         else:
#             print('using terabyte dataset for cache simulation... train_data should have been passed by parameter...')
          

#     print("Temporal Locality Simulation")
#     for vec_size in [64, 128, 256, 512]:
#         for cache_size in [1, 2, 4, 8]:
#             CacheSimulator(
#                 train_data=train_data,
#                 collision=collision,
#                 vec_size=vec_size,
#                 Cache_MB=cache_size,
#                 Block_size=64
#             )

#     print("Spatial Locality Simulation")
#     for vec_size in [64, 128, 256, 512]:
#         for cache_block_size in [64, 128, 256, 512, 1024]:
#             CacheSimulator(
#                 train_data=train_data,
#                 collision=collision,
#                 vec_size=vec_size,
#                 Cache_MB=1,
#                 Block_size=cache_block_size
#             )

# def get_random_trace(block_size=64, collision=8):
#     GB_TO_BYTES = 1024 * 1024 * 1024
#     TOTAL_SIZE_BYTES = 0.01 * GB_TO_BYTES #// collision
#     BLOCK_SIZE_BYTES = block_size

#     total_blocks = TOTAL_SIZE_BYTES // BLOCK_SIZE_BYTES
#     random_block = random.randint(0, total_blocks - 1)
#     random_address = random_block * BLOCK_SIZE_BYTES

#     return random_address

# def RunRandomCacheSimulation(train_data=None, collision=8, vec_size=64, Cache_MB=1, Block_size=64):
#     CACHE_SIZE = Cache_MB * 2**20  # in bytes
#     BLOCK_SIZE = Block_size  # in bytes
#     ASSOCIATIVITY = 4
#     cache = Cache(CACHE_SIZE, BLOCK_SIZE, ASSOCIATIVITY)
#     total_burst = vec_size // 64
#     total_iterations = 100000*26

#     for i in range(total_iterations):
#         q_pa = get_random_trace(Block_size)
#         for j in range(total_burst):
#             cache.access(q_pa + 64*j, 'q')

#     q_hitrate = round(cache.category_hit_rate('q'),3)
#     total_hitrate = round(cache.overall_hit_rate(),3)

#     print(f"\nRandom Cache Simulation on {Cache_MB}MB Cache / {Block_size}B block / Vector {vec_size}B / collision {collision}\n")
#     print(f"Q hitrate : {q_hitrate}")
#     print(f"Total hitrate : {total_hitrate}")



# def convertDictGenerator():
#     convertDicts = [{} for _ in range(26)]  # Initialize empty dictionaries for each categorical feature
#     npzfile = "./terabyte_input/day"
#     counts = np.zeros(26, dtype=np.int32)
#     for i in range(24):
#         print(f'Loading day {i} data from {npzfile}_{i}.npz...')
#         with np.load(f'./terabyte_input/day_{i}.npz') as data:
#             # Assuming X_cat_t is the transposed categorical data
#             X_cat_t = data["X_cat_t"]
#             for j in range(26):  # Iterate over each categorical feature
#                 unique_values = np.unique(X_cat_t[j, :])  # Find unique values for each feature
#                 for value in unique_values:
#                     convertDicts[j][value] = 1  # Update convertDicts

#     # Convert indices in convertDicts to sequential numbers
#     for j in range(26):
#         unique_values = list(convertDicts[j].keys())
#         convertDicts[j] = {old: new for new, old in enumerate(unique_values)}
#         dict_file_j = "day_fea_dict_{0}_tmp.npz".format(j)
#         if not os.path.exists(dict_file_j):
#             np.savez_compressed(
#                 dict_file_j,
#                 unique=np.array(list(convertDicts[j]), dtype=np.int32)
#             )
#         print("Saved convertDicts to file.")
#         counts[j] = len(convertDicts[j])

#     count_file = "day_fea_count_tmp.npz"
#     if not os.path.exists("./terabyte_input/day__fea_count_tmp"):
#         np.savez_compressed(count_file, counts=counts)





# if __name__ == "__main__":
#     RunRandomCacheSimulation(Cache_MB=1)
#     RunRandomCacheSimulation(Cache_MB=2)
#     RunRandomCacheSimulation(Cache_MB=4)
#     RunRandomCacheSimulation(Cache_MB=8)

#     # RunCacheSimulation()


#     # if not path.exists(total_file):
#     #     np.savez_compressed(total_file, total_per_file=total_per_file)
