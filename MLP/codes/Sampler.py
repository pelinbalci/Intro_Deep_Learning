
from torch.utils.data.sampler import SubsetRandomSampler

list_a = [0.1, 0.9, 0.4, 0.7, 3.0, 0.6]
sample_object = SubsetRandomSampler(list_a)
sample_list = list(SubsetRandomSampler(list_a))

# sample_object: <torch.utils.data.sampler.SubsetRandomSampler object at 0x10e4c3710>
# sample_list: <class 'list'>: [0.7, 0.4, 0.1, 0.6, 3.0, 0.9]

print('done')