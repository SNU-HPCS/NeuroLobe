import h5py
filename = "./spykingcircus_output/sorter_output/recording/recording.synthetic_templates.hdf5"
print(filename)

with h5py.File(filename, "r") as f:
    # Print all root level object names (aka keys) 
    # these can be group or dataset names 
    print("Keys: %s" % f.keys())

    # CODE FOR DATASETS (templates)
    for i in range(len(list(f.keys()))):
        # get first object name/key; may or may NOT be a group
        a_group_key = list(f.keys())[i]
        print("\n===============", a_group_key, "===============")

        # get the object type for a_group_key: usually group or dataset
        print(type(f[a_group_key])) 

        # If a_group_key is a group name, 
        # this gets the object names in the group and returns as a list
        # If a_group_key is a dataset name, 
        # this gets the dataset values and returns as a list
        data = list(f[a_group_key])

        print(data)
        # print(f[a_group_key])

        # preferred methods to get dataset values:
        ds_obj = f[a_group_key]      # returns as a h5py dataset object
        ds_arr = f[a_group_key][()]  # returns as a numpy array

        print(ds_arr)
        print(ds_arr.shape)

    # # CODE FOR GROUPS (result)
    # for group in f.keys() :
    #     print("\n===============", group, "===============")
    #     flag = False
    #     for dset in f[group].keys():      
    #         print (dset)
    #         ds_data = f[group][dset] # returns HDF5 dataset object
    #         # print (ds_data)
    #         # print (ds_data.shape, ds_data.dtype)
    #         arr = f[group][dset][:] # adding [:] returns a numpy array
    #         print (arr.shape, arr.dtype)
    #         if not flag:
    #             print (arr)
    #             flag=True