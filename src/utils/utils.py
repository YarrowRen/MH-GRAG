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