import random
import string
def generate_random_string(length=8):
    """
    生成一个指定长度的随机字符串，用于文件命名。
    
    参数:
    length (int): 随机字符串的长度。
    
    返回:
    str: 生成的随机字符串。
    """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

def communities_to_labels(communities, num_nodes):
    """
    将社区分配转换为节点标签列表。
    
    参数:
    communities (list of list of int): 社区列表，每个社区是节点的列表。
    num_nodes (int): 图中节点的总数。
    
    返回:
    list of int: 节点标签列表，其中每个索引对应一个节点，值为该节点所属的社区标签。
    """
    labels = [-1] * num_nodes
    for label, community in enumerate(communities):
        for node in community:
            labels[node] = label
    return labels
