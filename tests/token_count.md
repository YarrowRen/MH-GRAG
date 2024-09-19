# 以explagraph为例

- 其包含12k实体以及10k关系
- 使用Leiden聚类后会生成约100个社区以及600个子社区

---

1. 生成实体和关系的description阶段，每个实体大约消耗10k Token
2. 生成社区和子社区的summary阶段，每个社区大约消耗50k Token

假设在描述生成阶段和社区/子社区生成阶段的输入和输出 token 的比例为 **1:1**，即输入 token 和输出 token 各占消耗的一半。分别基于 Doubao-lite-32k 和 GPT-4o-mini 的定价进行费用计算。


### 总消耗量计算：

#### 1. **生成实体和关系的 description**：
   - 22k 实体和关系，每个消耗 **10k tokens**。
   $$
   22,000 \, \text{entities/relations} \times 10,000 \, \text{tokens/entity} = 220,000,000 \, \text{tokens}
   $$
   - **输入 token**：110,000,000 tokens
   - **输出 token**：110,000,000 tokens

#### 2. **生成社区和子社区的 summary**：
   - 700 个社区和子社区（100 社区 + 600 子社区），每个消耗 **50k tokens**。
   $$
   700 \, \text{communities/sub-communities} \times 50,000 \, \text{tokens/community} = 35,000,000 \, \text{tokens}
   $$
   - **输入 token**：17,500,000 tokens
   - **输出 token**：17,500,000 tokens

### 总消耗量：
- **输入 tokens**：110,000,000 + 17,500,000 = **127,500,000 tokens**
- **输出 tokens**：110,000,000 + 17,500,000 = **127,500,000 tokens**
- **总 token 消耗量**：127,500,000 + 127,500,000 = **255,000,000 tokens**

---

### Doubao-lite-32k 定价（人民币）：

1. **输入 token** 消耗：
   $$
   \frac{127,500,000 \, \text{tokens}}{1,000} \times 0.0003 \, \text{元/千 tokens} = 38.25 \, \text{元}
   $$
   
2. **输出 token** 消耗：
   $$
   \frac{127,500,000 \, \text{tokens}}{1,000} \times 0.0006 \, \text{元/千 tokens} = 76.5 \, \text{元}
   $$

3. **总费用（人民币）**：
   $$
   38.25 \, \text{元} + 76.5 \, \text{元} = 114.75 \, \text{元}
   $$

---

### GPT-4o-mini 定价（美元）：

1. **输入 token** 消耗：
   $$
   \frac{127,500,000 \, \text{tokens}}{1,000,000} \times 0.150 \, \text{美元/百万 tokens} = 19.125 \, \text{美元}
   $$
   
2. **输出 token** 消耗：
   $$
   \frac{127,500,000 \, \text{tokens}}{1,000,000} \times 0.600 \, \text{美元/百万 tokens} = 76.5 \, \text{美元}
   $$

3. **总费用（美元）**：
   $$
   19.125 \, \text{美元} + 76.5 \, \text{美元} = 95.625 \, \text{美元}
   $$

---

### 结果总结：
- **Doubao-lite-32k 总费用**：**114.75 元**
- **GPT-4o-mini 总费用**：**95.625 美元**