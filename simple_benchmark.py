import torch
import torch.nn.functional as F

#! This one is the naive torch based spVspM
def llm_spv_spm_th_based(x, w, th):

    #! Assume the x sent in is initially dense.
    mask = torch.abs(x) >= th
    x = x * mask

    #! x should have shape of [1,1,hidden]
    output = F.linear(x,w) #! we assume w is already sparse.

    return output


def llm_spv_spm_sort_based(x,w, sparsity=0.5):

    thresh = torch.sort(torch.abs(x), dim=-1)[0][:, :, int(x.shape[-1] * sparsity)].unsqueeze(-1)
    mask = torch.abs(x) >= thresh
    x = x * mask

    #! x should have shape of [1,1,hidden]
    output = F.linear(x,w) #! we assume w is already sparse.

    return output


def test():

    torch.manual_seed(0)
    
    x_shape = 4096
    w_shape = 4096
    x_spa = 0.5
    w_spa = 0.5

    x = torch.randn(1, 1, x_shape)
    thresh = torch.sort(torch.abs(x), dim=-1)[0][:, :, int(x.shape[-1] * x_spa)].unsqueeze(-1)

    weight = torch.randn(w_shape, w_shape)
    total_elements = weight.numel()
    num_zeros = int(w_spa * total_elements)
        
    mask = torch.ones_like(weight).flatten()
    zero_indices = torch.randperm(total_elements)[:num_zeros]
    mask[zero_indices] = 0
    mask = mask.reshape(weight.shape)
    
    weight = weight * mask

    #! threshold based method
    output_th = llm_spv_spm_th_based(x,weight,thresh)
    print(output_th)

    #! on-the-fly sorting based
    output_sort = llm_spv_spm_sort_based(x,weight,x_spa)
    print(output_sort)




if __name__ == '__main__':
    test()