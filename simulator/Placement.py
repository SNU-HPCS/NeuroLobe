from math import log2, ceil
import GlobalVars as GV
import numpy as np
import copy

# NOTE:
# Our compiler utilizes the iterative data access pattern of the BCI workloads
# to minimize the on-chip memory fragmentation
# 1) Replaces the data position
# 2) Merges the cells and add paddings for an accurate data addressing

class Bucket:
    def __init__(self, budget, precision_list, valid_list):
        self.indices = []
        max_precision, max_index = self.get_max(precision_list, valid_list)
        self.budget = budget - max_precision
        self.indices.append(max_index)
        valid_list[max_index] = False

    def insert(self, precision_list, valid_list):
        for prec in sorted(precision_list, reverse = True):
            # different precision
            if self.budget - prec >= 0:
                index = precision_list.index(prec)
                # check if the target is valid
                if valid_list[index]:
                    self.budget -= prec
                    self.indices.append(index)
                    valid_list[index] = False
                    return True
        return False

    def get_max(self, precision_list, valid_list):
        max_dat = -1
        max_ind = -1
        for ind in range(len(precision_list)):
            if max_dat <= precision_list[ind] and valid_list[ind]:
                max_dat = precision_list[ind]
                max_ind = ind
        return max_dat, max_ind

# We should perform a fragmentation-aware data placement
def combine_entry(entry_info):
    if not entry_info: return [], {}, 0, {}
    # Extract the precision list for the bucketing
    precision_list = []
    for entry in entry_info:
        precision_list.append(entry['prec'])

    min_budget = ceil(log2(max(precision_list)))
    # The maximum budget within a single entry should be less or equal to the memory width
    max_budget = min(ceil(log2(sum(precision_list))), ceil(log2(GV.MEM_WIDTH['corr_mem'])))
    # There exist two different cases
    # 1) The whole data for each iteration fit within a single line
    #    => We do not put additional padding
    # 2) The whole data for each iteration does not fit within a single line


    # Add padding and replace the data position
    # +) add offset to use in the compiler
    aligned_entry_info = []
    aligned_offset_info = {}
    aligned_loop_offset = 0
    aligned_width_info = {}
    data_offset = 0

    min_cost = float('inf')
    min_bucket = None
    target_budget = None
    for budget in range(min_budget, max_budget + 1):
        budget_dat = 2 ** budget
        cost, bucket_list = calc_cost(precision_list, budget_dat)

        cost = budget_dat * len(bucket_list)
        if min_cost > cost:
            min_cost = cost
            min_bucket = bucket_list
            target_budget = budget_dat
    # If the whole budget is less than the inter-layer optimization
    # => We can use smaller value without padding
    for bucket in min_bucket:
        indices = bucket.indices
        padding = bucket.budget
        for ind in indices:
            aligned_entry = entry_info[ind]
            aligned_offset_info[entry_info[ind]['name']] = data_offset
            aligned_entry['original'] = ind
            data_offset += aligned_entry['prec']
            # We check address jump at the start of the line
            aligned_entry_info.append(aligned_entry)
        # Add padding to the last entry in the bucket
        aligned_entry_info[-1]['prec'] += padding
        data_offset += padding
        for ind in indices:
            aligned_width_info[entry_info[ind]['name']] = aligned_entry['prec']
    aligned_loop_offset = data_offset

    return aligned_entry_info, aligned_offset_info, aligned_loop_offset, aligned_width_info

def calc_cost(precision_list, budget_dat):
    precision_temp = copy.deepcopy(precision_list)
    valid_list = [True for _ in range(len(precision_temp))]

    # start with an initial budget list
    bucket_list = [Bucket(budget_dat, precision_temp, valid_list)]
    cost = 0
    while not all(not valid for valid in valid_list):
        bucket = bucket_list[-1]
        if not bucket.insert(precision_temp, valid_list):
            bucket_list.append(Bucket(budget_dat, precision_temp, valid_list))

    cost = budget_dat * len(bucket_list)
    return cost, bucket_list

if __name__ == "__main__":
    precision_array = [7,5,3,10,1,10]
    cost, bucket_list = combine_entry(precision_array)
    print(precision_array)
    print(cost)
    for bucket in bucket_list:
        print(bucket.indices)
