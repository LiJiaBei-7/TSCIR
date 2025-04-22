import torch

def plackett_luce_ranking(similarity_matrix):
    # 假设 similarity_matrix 是一个 (bsz, bsz) 的相似度矩阵
    bsz = similarity_matrix.size(0)
    
    # 复制 similarity_matrix，以便在采样过程中修改
    scores = similarity_matrix.clone()
    
    # 用于存储每个样本的排名
    rankings = torch.zeros(bsz, bsz, dtype=torch.long)
    
    # 初始可用索引列表
    available_indices = torch.arange(bsz).unsqueeze(0).expand(bsz, -1)
    
    # 初始化一个 tensor 来存储累积和
    cum_sum_scores = scores.sum(dim=1, keepdim=True)
    
    # 逐步选择排名
    for i in range(bsz):
        # 计算每个样本的概率
        probs = scores / cum_sum_scores
        
        # 采样：从每个样本的概率分布中选择一个索引
        chosen_indices = torch.multinomial(probs, 1).squeeze()
        
        # 记录排名
        rankings[:, i] = available_indices[torch.arange(bsz), chosen_indices]
        
        # 设置已选元素为 -inf 以确保在下一次计算时不会被选中
        scores[torch.arange(bsz), chosen_indices] = -float('inf')
        
        # 更新 cum_sum_scores 以反映移除后的 scores
        cum_sum_scores = scores[scores != -float('inf')].view(bsz, -1).sum(dim=1, keepdim=True)
        
        # 使用布尔掩码更新 available_indices
        mask = torch.ones_like(available_indices, dtype=torch.bool)
        mask[torch.arange(bsz), chosen_indices] = False
        available_indices = available_indices[mask].view(bsz, -1)

    return rankings

# 示例相似度矩阵
bsz = 4
similarity_matrix = torch.tensor([
    [1.0, 0.8, 0.5, 0.3],
    [0.8, 1.0, 0.6, 0.4],
    [0.5, 0.6, 1.0, 0.2],
    [0.3, 0.4, 0.2, 1.0]
])

rankings = plackett_luce_ranking(similarity_matrix)
print("Rankings:\n", rankings)