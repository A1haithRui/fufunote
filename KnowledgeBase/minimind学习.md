## Pretrain
在dataset中，定义了预训练的数据类型。代码如下：
```python
class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=data_path, split='train')
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        sample = self.samples[index]
        # 构建输入文本
        encoding = self.tokenizer(
            str(sample['text']),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding.input_ids.squeeze()
        loss_mask = (input_ids != self.tokenizer.pad_token_id)

        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)
        return X, Y, loss_mask
```
在pretrain的数据(pretrain_hq.jsonl)中，可以看出每一行的结构大概是：
```json
{"text": "鉴别一组中文文章的风格和特点，例如官方、口语、文言等。需要提供样例文章才能准确鉴别不同的风格和特点。<|im_end|> 好的，现在帮我查一下今天的天气怎么样?今天的天气依据地区而异。请问你需要我帮你查询哪个地区的天气呢？<|im_end|> 打开闹钟功能，定一个明天早上七点的闹钟。好的，我已经帮您打开闹钟功能，闹钟将在明天早上七点准时响起。<|im_end|> 为以下场景写一句话描述：一个孤独的老人坐在公园长椅上看着远处。一位孤独的老人坐在公园长椅上凝视远方。<|im_end|> 非常感谢你的回答。请告诉我，这些数据是关于什么主题的？这些数据是关于不同年龄段的男女人口比例分布的。<|im_end|> 帮我想一个有趣的标题。这个挺有趣的：\"如何成为一名成功的魔术师\" 调皮的标题往往会吸引读者的注意力。<|im_end|> 回答一个问题，地球的半径是多少？地球的平均半径约为6371公里，这是地球自赤道到两极的距离的平均值。<|im_end|> 识别文本中的语气，并将其分类为喜悦、悲伤、惊异等。\n文本：“今天是我的生日！”这个文本的语气是喜悦。<|im_end|>"}
...
```
### 1. 文本向量化(Tokenization)
就是代码中的`encoding = self.tokenizer(...)`，这个分词器将数据中那些带有 `<|im_end|>` 的字符串切分成 Token，并映射为整数 ID。其中，`padding = 'max_length'`表示，如果文本长度短于设定值(一般是512)，就用`pad_token_id`来补齐(一般用0)；`truncation = True`表示，如果文本太长，直接截断；这样每个Batch里的数据形状都是统一的`max_length`。
>这里的每个id处并不是一个字，也不是一个词，而是一个token。现在模型大多用BPE(Byte Pair Encoding)或类似算法来做子词切分。这里an的文档有记载。
```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("./model")
text1, text2 = "我爱吃炒面", "我爱玩原神"
tokens1, tokens2 = tokenizer.encode(text1), tokenizer.encode(text2)
print(f"ID序列: tokens1:{tokens1}; tokens:{tokens2}")
print(f"还原: text1:{[tokenizer.decode([t]) for t in tokens1]}, text2:{[tokenizer.decode([t]) for t in tokens2]}")
# 最终输出
>>>ID序列: tokens1:[397, 1699, 3442, 3419, 583]; tokens:[397, 1699, 2237, 1484, 2031]
>>>还原: text1:['我', '爱', '吃', '炒', '面'], text2:['我', '爱', '玩', '原', '神']
```
>可以通过上述代码做个小测试
### 2. "词语接龙"(X & Y)
那模型该怎么学习说话呢？就是这里定义的`X`和`Y`。模型的目标**是给定前 $i$ 个词，预测第 $i+1$ 个词**。**X(Input)**：切掉了最后一个词。模型看到的是从第0个到倒数第n-1个词。 **Y(Label)**：切掉了第一个词。模型需要预测的目标是从第 1 个到n-1个词。
举个例子，原始id是[0, 1, 2, 3, 4, 5]，长度为5，那么输入X的id就是[0, 1, 2, 3, 4]，输出Y的id就是[1, 2, 3, 4, 5]。
### 3. 掩码损失(loss mask)
在构建X和Y之前就定义了`loss_mask = (input_ids != self.tokenizer.pad_token_id)`，因为我们对文本做了`padding`，为了补齐长度而填充的token是没有实际意义的，我们不希望模型学习如何预测这些占位符，所以就将它们记录并mask掉。
某处如果不是填充字符，`loss_mask`就是1，参与计算loss；否则就是0，不参与计算loss。
而最后也对`loss_mask`做了[1:]的偏移处理(`loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)`)，这样刚好和模型输出Y在位置上完美对应。
## SFT
SFT的代码和pretrain的很像。
```python
class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids
    def __len__(self):
        return len(self.samples)
    def create_chat_prompt(self, cs):
        ...
    def generate_loss_mask(self, input_ids):
        ...
    def __getitem__(self, index):
        sample = self.samples[index]
        prompt = self.create_chat_prompt(sample['conversations'])
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        loss_mask = self.generate_loss_mask(input_ids)
        # 构建训练数据
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # 对齐预测位置
        # # === 打印每个token的掩码情况 ===
        # print(f"\n--- Sample {index} ---")
        # for i, (x, y, m) in enumerate(zip(X, Y, loss_mask)):
        #     print(f"{i:3d}: X={self.tokenizer.decode([x])!r:16s} ---> Y={self.tokenizer.decode([y])!r:16s} mask={m}")
        # # ================================
        return X, Y, loss_mask
```
在初始化的时候定义了`bos_id`和`eos_id`，分别'begin of string'和'end of string'，因为sft现在要有角色扮演和对话，所以需要指定一套协议。`bos_id`相当于告诉模型，现在要开始一段新的指令或回答；`eos_id`是告诉模型预测到这处id时就停止生成，不要为了继续凑字数而胡言乱语。
```json
{"conversations": [{"role": "user", "content": "请告诉我在中国古代的“四大发明”是什么？"}, {"role": "assistant", "content": "中国古代的“四大发明”是指造纸术、印刷术、火药和指南针。这四项发明对世界文明的发展产生了深远的影响：\n\n1. **造纸术**：据史书记载，东汉时期的蔡伦改进了造纸工艺，使得纸张的生产更加便捷、成本更低，质量也更加优良。这一发明极大地促进了文化的传播和保存。\n\n2. **印刷术**：中国古代的印刷术最早可以追溯到唐代的雕版印刷，到了宋代发展出了活字印刷技术。印刷术的发明极大地促进了知识的传播，降低了书籍的成本，对教育和文化的发展起到了重要的推动作用。\n\n3. **火药**：火药最初是在唐代被发现的，最初可能用于医疗或炼金术。到了宋代，人们开始将火药用于军事目的，发明了各种火器。火药的发明改变了战争的面貌，对世界军事技术的发展产生了重大影响。\n\n4. **指南针**：指南针最初在中国被用于风水测量，后来逐渐发展成为航海导航的重要工具。这一发明极大地促进了海上航行技术的发展，对于新航路的开辟和世界地理大发现起到了关键作用。\n\n这四项发明不仅在中国历史上占有重要地位，而且对全世界的科技进步和文明发展都产生了深远的影响。"}]}
...
```
从数据这里来看，`bos_id`紧接`assistant`代表着该字符串将开始，并切换角色为"assistant"。
### 1. 创建聊天提示词(create_chat_prompt)
预训练时直接喂了文本，但是SFT时模型需要知道谁在说话，所以需要找角色。我们把文本对话"conversation"(即`cs`)输入进去，里面包含了`"role"`(如 `user` 或 `assistant`)和`content`列表。然后根据模型定义的 `tokenizer_config.json`，把列表转成特定格式。不同的模型(比如Llama或者Qwen)会有不同的特殊token序列，所以要应用不同的模板。
```python
def create_chat_prompt(self, cs):
    messages = cs.copy()
    # 提取函数调用工具（如果有的话）
    tools = cs[0]["functions"] if (cs and cs[0]["role"] == "system" and cs[0].get("functions")) else None
    # 核心：调用 Tokenizer 的模板引擎
    return self.tokenizer.apply_chat_template(
        messages,
        tokenize=False,            # 先不转成数字，先生成完整的字符串
        add_generation_prompt=False, # 训练时不需要在末尾强行加 "assistant" 的引导符
        tools=tools
    )
```
转换前：`{"role": "user", "content": "你好"}`，
转换后：`<|im_start|>user\n你好<|im_end|>\n`（这是 MiniMind 模仿 Qwen 的常见格式）。
### 2. 生成掩码损失(generate_loss_mask)
和预训练的略有差异。预训练时的掩码是为了遮住填充的占位符，而SFT的掩码是为了定位回答中的有效位置。
```python
def generate_loss_mask(self, input_ids):
    loss_mask = [0] * len(input_ids) # 默认全为 0，即默认不计算 Loss
    i = 0
    while i < len(input_ids):
        # 1. 寻找回答的开始：检测序列是否出现了 self.bos_id (即 <|im_start|>assistant)
        if input_ids[i:i + len(self.bos_id)] == self.bos_id:
            start = i + len(self.bos_id) # 确定回答内容的第一个 token 位置
            end = start
            # 2. 寻找回答的结束：向后扫描，直到遇见 self.eos_id (即 <|im_end|>)
            while end < len(input_ids):
                if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                    break
                end += 1
            # 3. 标记 Loss：将回答部分的坐标在 mask 中设为 1
            # 注意：start + 1 是因为 Y 序列是 X 序列偏移一位的结果，我们要盖住的是 Y 中的第一个预测词
            for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                loss_mask[j] = 1
            # 跳过已处理部分
            i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
        else:
            i += 1
    return loss_mask
```
本质上就是在所有的token id中找有效的位置。既要找到`bos_id`和`eos_id`之间的位置来定位回答开始，还要跳过提问中的内容来防止模型伪造用户提问。有效的区域只在`assistant`之后，`im_end`之前的这个闭区间，模型预测有结果后才会计算损失回传梯度。
>至于__getitem__，和预训练的逻辑基本一致，只是这里要创建聊天提示词，生成有效区域的掩码损失。
## LoRA
LoRA是微软提出的一种高效的参数微调方法(Parameter-Efficient Fine-Tuning, PEFT)，旨在通过将原本权重进行低秩分解的方式来对预训练模型进行微调。相比于全参数微调(Full Fine-Tuning)，LoRA只需要更新少量参数。假设模型更新的增量$\Delta W(1000 \times 1000)$是低秩的，LoRA将其分解为两个小矩阵$B(1000 \times 8)$和$A(8\times 1000)$的乘积，这样$W =  W_0 + \Delta W = W_0 + BA$，可以大幅减少训练量。原始的模型权重$W_0$在训练期间保持冻结，仅训练$A$和$B$。
### 1. LoRA结构
```python
class LoRA(nn.Module): # 定义Lora网络结构
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank  # LoRA的秩（rank），控制低秩矩阵的大小
        self.A = nn.Linear(in_features, rank, bias=False)  # 低秩矩阵A
        self.B = nn.Linear(rank, out_features, bias=False)  # 低秩矩阵B
        self.A.weight.data.normal_(mean=0.0, std=0.02) # 矩阵A高斯初始化
        self.B.weight.data.zero_() # 矩阵B全0初始化
    def forward(self, x):
        return self.B(self.A(x))
```
这里的`self.A`是低秩的降维矩阵，`self.B`是低秩的升维矩阵；前者高斯初始化是为了让初始状态有梯度，后者全0初始化是为了让训练的第一步$B\times A=0$，这样模型刚开始训练时，LoRA部分对原模型的影响为0，模型从原始权重开始平滑微调。如果B也随机初始化，那么模型一开始就摒弃了旧权重的能力，开始胡言乱语。
### 2. 如何使用LoRA？
```python
def apply_lora(model, rank=8):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(model.device)
            setattr(module, "lora", lora)
            original_forward = module.forward
            # 显式绑定
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)
            module.forward = forward_with_lora
```
首先，LoRA只会挂在形状对称的线性层(通常是Q, K, V, O矩阵)，所以一开始要`isinstance()`一下。然后用`setattr`来给他们强行塞入一个属性`lora`。
然后拦截原有的`forward`，加入补丁`forward = forward_with_lora`，返回`original_forward(x) + lora(x)`，这样就对应了公式$y = W_0x + \Delta W x$
```python
def load_lora(model, path):
    state_dict = torch.load(path, map_location=model.device)
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
            module.lora.load_state_dict(lora_state)
def save_lora(model, path):
    state_dict = {}
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            clean_name = name[7:] if name.startswith("module.") else name
            lora_state = {f'{clean_name}.lora.{k}': v for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)
```
在存储的时候，遍历整个模型，只找`hasattr`带有`'lora'`属性的层，忽略原本模型的巨大权重，只把$A$和$B$的参数挑出来存进`state_dict`中。而为了防止分布式训练(DDP)的时候要在参数名前面加上`model.`，所以这里用了`k.replace`和`k[7:]`来消掉前面的参数名。
## Refine Learning
大语言模型里的强化学习模型可以分为两类。$\begin{cases} RLHF: 基于人类反馈 \\ RLAIF: 基于AI反馈\end{cases}$。
不管是那一类，整个框架都设计了四个模型来协作。这四个模型分别是Actor(演员)、Crtic(评论员)、Reward(裁判)和Reference(参考)。
Actor会生成token(相当于模型输出)，Critic预测Actor说的这句话能得多少分，Reward根据偏好给最终结果打分，Reference则是原始的SFT模型，约束结果，防止跑偏。这个就是最经典的Actor-Critic结构。
### DPO(直接偏好优化)
DPO就是一个经典的、不需要显式奖励模型的离线RLHF。
在之前的训练和微调中，模型已经具备了对话能力并且适配对应的语境，但是它不知道什么样的回答是好的，什么样的回答是坏了，我们没有给模型正反样例去激励它认清好坏。为了让模型能够符合我们人类的偏好，我们就给模型举一些好的例子和坏的例子，降低模型回答出不好的回答的产生概率。
#### 数据格式
这里就拿`dpo.jsonl`里最后一行的数据举个例子，他会给两个对话，前者为`chosen`表示好的回答，后者为`rejected`表示不好的回答。
```python
{"chosen": 
	[{"content": "我想让你和我就这个命题进行争论：“人类的美好时光在未来。”我希望你支持这个命题进行争论。", "role": "user"}, {"content": "绝对，我非常乐意支持这个命题：“人类的美好时光在未来”。\n\n1. 技术的持续进步：过去几十年来，技术进步的速度惊人地快，导致了医疗保健、通讯、交通等各个方面的改善。这种进步没有放缓的迹象，......(此处省略若干字)，努力迈向更美好的世界，我们就能确保我们的未来比过去更加光明。", "role": "assistant"}], 
"rejected": 
	[{"content": "我想让你和我就这个命题进行争论：“人类的美好时光在未来。”我希望你支持这个命题进行争论。", "role": "user"}, {"content": "当然，我很乐意支持“人类最美好的日子在未来”的观点。让我提出一些支持这个想法的观点：\n\n1. 技术进步：对人类未来感到......(此处省略若干字)，让我们有理由相信人类最美好的日子在未来。", "role": "assistant"}]}
```
而对应的数据类型定义在dataset中，代码为：
```python

class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids
        self.data = load_dataset('json', data_files=file_path, split='train')
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        ...
    def generate_loss_mask(self, input_ids):
        ...
```
和预训练与SFT基本相同，只是初始化的时候还多了一个json格式的加载。
-  1. 双路并行处理数据(chosen & rejected)
相比与预训练和SFT只有X和Y两种数据格式，SFT将数据划分为成两个角色(user和assistant)，而DPO在assistant的基础上还划分了`chosen`和`rejected`。让模型为一个问题同时处理两套输入(x)和标签(y)，并且设计合适的mask来定位位置。
```python
def __getitem__(self, index):
	item = self.data[index]
	chosen = item['chosen']  # 是一个 list，里面包含若干 {role, content}
	rejected = item['rejected']  # 同上
	chosen_prompt = self.tokenizer.apply_chat_template(
		chosen, tokenize=False, add_generation_prompt=False
	)
	rejected_prompt = self.tokenizer.apply_chat_template(
		rejected, tokenize=False, add_generation_prompt=False
	)
	chosen_encoding = self.tokenizer(
		chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
	)
	rejected_encoding = self.tokenizer(
		rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
	)
	chosen_input_ids = chosen_encoding['input_ids']
	chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)
	rejected_input_ids = rejected_encoding['input_ids']
	rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)
	x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
	y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
	mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
	x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
	y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
	mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)
	return {
		'x_chosen': x_chosen,
		'y_chosen': y_chosen,
		'mask_chosen': mask_chosen,
		'x_rejected': x_rejected,
		'y_rejected': y_rejected,
		'mask_rejected': mask_rejected
	}
```
- 2. mask定位和屏蔽
这里的mask和sft与预训练基本一直，都是为了定位有效信息的位置。
```python
def generate_loss_mask(self, input_ids):
	loss_mask = [0] * len(input_ids)
	i = 0
	while i < len(input_ids):
		if input_ids[i:i + len(self.bos_id)] == self.bos_id:
			start = i + len(self.bos_id)
			end = start
			while end < len(input_ids):
				if input_ids[end:end + len(self.eos_id)] == self.eos_id:
					break
				end += 1
			for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
				loss_mask[j] = 1
			i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
		else:
			i += 1
	return loss_mask
```
#### 训练
首先呢，我们要把概率正确的提出出来，假设对应的`logits`概率是$P$，那么我们要先求$\log{P}$，因为对数在数学上比较稳定，可以防止连乘导致的数值下溢。然后再用一个`torch.gather`来在预测的数万个词的概率中，找标签里的词的概率，然后把它提取出来，对应代码为：
```python
def logits_to_log_probs(logits, labels):
    # logits shape: (batch_size, seq_len, vocab_size)
    # labels shape: (batch_size, seq_len)
    # log_probs shape: (batch_size, seq_len)
    log_probs = F.log_softmax(logits, dim=2)
    log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return log_probs_per_token
```
然后就是经典的损失公式$Loss = -\log \sigma (\beta (\log \frac{\pi_{\theta}(y_w|x)}{\pi_{ref}(y_w|x)} - \log \frac{\pi_{\theta}(y_l|x)}{\pi_{ref}(y_l|x)}))$，~~这里有三项，分别是策略项、优势项和正则项。策略项($f(r_t)=\log{⁡r_w}−\log⁡{r_l}$)负责对比chosen和rejected的概率比；优势项($g(A_t)=|$)来进行偏好对比，避免显式计算；正则项($h(\mathrm{KL}_t)= \beta (\cdot)$)隐含在$\beta$中来控制偏离参考模型的程度。~~(Gemini说这样表述不好)
$\beta \log \frac{\pi_{\theta}}{\pi_{ref}}$ 这一项代表了**隐式奖励（Implicit Reward）**。Loss函数通过对比$Chosen$的奖励和$Rejected$的奖励差，利用`logsigmoid`引导模型偏向人类偏好。$\beta$起到了**KL惩罚系数**的作用，防止策略模型$\pi_{\theta}$偏离参考模型$\pi_{ref}$太远。
```python
def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    # ref_log_probs 和 policy_log_probs 都是 shape: (batch_size, seq_len)
    seq_lengths = mask.sum(dim=1, keepdim=True).clamp_min(1e-8)  # 防止零长度mask导致除零NaN
    ref_log_probs = (ref_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()
    policy_log_probs = (policy_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()
    # 将 chosen 和 rejected 数据分开
    batch_size = ref_log_probs.shape[0]
    chosen_ref_log_probs = ref_log_probs[:batch_size // 2]
    reject_ref_log_probs = ref_log_probs[batch_size // 2:]
    chosen_policy_log_probs = policy_log_probs[:batch_size // 2]
    reject_policy_log_probs = policy_log_probs[batch_size // 2:]
    pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()
```
这里的`ref_log_probs`和`policy_log_probs`都和`mask`相乘再除以序列长度`seq_lengths`是为了计算参考句子和actor输出的句子的每个token的平均对数概率(如果不除以长度，长句子天然会比短句子拥有更小的总概率，因为有更多的负数相加，导致模型更喜欢短句子)。
那为什么当前策略模型输出要`batch_size`要`//2`呢？因为工程上假设`chosen`和`reject`是连续排列的，前一半是`chosen`，后一半是`reject`，这样就能比较好的把两者拆分出来。
`pi_logratios = chosen_policy_log_probs - reject_policy_log_probs`和`ref_logratios = chosen_ref_log_probs - reject_ref_log_probs`操作是为了计算逻辑比值。因为对数空间中减法相当于除法，两个`logratios`实际表示`chosen`比`rejected`要好多少，`pi`中的是当前策略模型actor输出句子，`ref`中的是SFT参考模型的句子。
`logits = pi_logratios - ref_logratios`计算了当前策略模型对参考模型的相对优势增量，如果当前模型拉开了`chosen`和`reject`的差距，那么这个值就越大。最后一个Sigmoid操作`loss = -F.logsigmoid(beta * logits)`将差距映射到0和1之间，因为要让差距越大越好，所以前面加了负号求最小化。`beta`就是控制模型对偏好信息的敏感度，是一个可调的超参，让模型不敢偏离原有的SFT参考模型太远。
然后简单看一下效果。模型从原来的“由于下雨建议去室内场所”变成了“由于下雨建议关注交通和实时天气”。这反映了DPO数据集中某种偏好（Preference）的注入，让模型更倾向于提供功能性建议而非生活方式建议。
```python
💬: 如果明天下雨，我应该如何出门
(sft输出)🤖: 出门前请先了解天气预报，以便做好相应的准备。可以去当地的天气预报网站，或者去当地的博物馆或美术馆看看。
(dpo输出)🤖: 出门前建议你先查看当地天气预报，或者联系当地的当地交通部门，查看最新的天气预报，以获取最新的天气信息。同时，你可以查看当地的天气预报网站或者应用以获取最新的天气情况。
```
>最后再说一句，DPO比PPO稳，因为它把策略优化问题转成了简单的分类问题，一会整理PPO的时候就会发现Actor-Critic架构要做非平稳的在线采样和复杂的超参调优。
### PPO(近端策略优化)
刚才的DPO使用人类标注`chosen`和`rejected`偏好对来做策略优化。这里我们完全由AI来担任裁判，这样就是一个RLAIF。
#### 奖励(Reward)
这里的AI裁判可以是model-base的奖励大模型(reward model)，也可以是像deepseek-R1一样设置规则函数进行校验，也可以是例如工具调用的环境反馈……用AI可以避免昂贵的人工标注，在线生成海量的训练数据，让模型在大量进化中快速进化。我们这里的PPO用的是一个小型但是高质量的InternLM2-1.8B-Reward的奖励模型作为reward，需要单独下载。
```python
pip install modelscope
modelscope download --model Shanghai_AI_Laboratory/internlm2-1_8b-reward
```
除了这个(InternLM2-1.8B-Reward)，还可以用通用的LLM+prompt来进行打分(如Qwen3-as-a-Judge)。
这里再补充一下，刚才说了AI裁判有三种，另外两种Rule-based和Environment-based这里就不详细记录，只记录和minimind有关的。
#### 数据准备
这里的RLAIF的数据集，是从SFT数据集中随机采样了1万条高质量对话，数据格式和SFT一致，但是因为训练过程会用策略模型来实时生成回答，所以assistant并不需要内容。所以格式大致如下：
```json
{"conversations": [
	{"role": "user", "content": "列出五个基本的人格理论，并分别以一句话概括。"}, 
	{"role": "assistant", "content": "空"}
]}
```
策略模型会基于user的问题生成1个或多个候选回答，然后由奖励模型来对回答进行打分，分数高的会被鼓励(增加$\Pi$策略)，分数低的回答会被抑制(降低$\Pi$策略)，然后循环，这也是强化学习的核心流程。
而这里的代码我就不详细分析，和SFT基本一致，只是把创建聊天提示词这里勾选为True，然后也不需要掩码损失了，因为我们现在不需要学说话(doge)。
```python
class RLAIFDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids
    def __len__(self):
        return len(self.samples)
    def create_chat_prompt(self, conversations):
        messages = []
        answer = ''
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content']})
            answer = turn['content']
        return self.tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True  # 这里需要True
        ), answer
    def __getitem__(self, index):
        sample = self.samples[index]
        prompt, answer = self.create_chat_prompt(sample['conversations'])
        return {
            'prompt': prompt,
            'answer': answer
        }
```
#### 演员和评论员(Actor和Critic)
前面不管是做DPO还是做LoRA，策略模型都是用SFT后的权重来做的。但是在PPO里，策略模型Actor就不能再选SFT模型，因为普通的SFT模型只学会了"对话"，它并没有学会"思考"。直接用它跑PPO，它根本吐不出`<think>`标签。我们需要训练一个推理SFT模型，来让模型学会思维链(CoT)。指路[[minimind学习#Reason]]。这里得到的Reason模型才是Actor。
那Critic呢？这里和前述的奖励模型还有些区别，Reward是用来打分的，它在PPO的过程中完全冻结，不参与参数更新；Critic本身还是由Actor(reason模型)来初始化，只不过在它的后面接了一个`value_head`，实时观察Actor生成的每一个词，并预测这个词及后续能带来的期望奖励值。
```python
class CriticModel(MiniMindForCausalLM):
    def __init__(self, params):
        super().__init__(params)
        # 替换lm_head为输出单一价值的线性层
        self.value_head = nn.Linear(params.hidden_size, 1)
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # 使用基础模型获取隐藏状态
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = self.model.norm(outputs[0])
        # 使用value_head获取价值估计
        values = self.value_head(hidden_states).squeeze(-1)
        return values
```
#### 训练
在PPO训练的代码中，需要专门计算奖励(calculate reward)，这里详细学习一下。
```python
def calculate_rewards(prompts, responses, reward_model, reward_tokenizer):
	def reasoning_model_reward(rewards): # 推理的格式奖励
        ...
		return rewards
    rewards = torch.zeros(len(responses), device=args.device)
    # 格式和标记奖励
    if args.reasoning == 1: # 如果开了推理模式，就要算格式和标记奖励
        rewards = reasoning_model_reward(rewards)
	# 使用reward model计算整个response的奖励
    with torch.no_grad():
	    ...
    return rewards
```
##### 1. 格式奖励(严格)
啥叫格式奖励呢？就是模型的输出完全符合标准的推理模板(有且仅有一对`<think>`标签，且顺序正确)，这样就给对应的reward加0.5分。
##### 2. 标记奖励
格式奖励要求完全符合模板，那假如写了一个`<think>`但是忘了闭合`</think>`，在严格模式下是拿不到分的，但是在这里可以拿到0.25分，为模型提供一个阶梯式反馈，引导它一步一步学会写出完整标签。
```python
	def reasoning_model_reward(rewards):
        # 1. 格式奖励（仅针对训练推理模型时使用）
        pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        pattern2 = r"^<think>\n.*?\n</think>\n\n<answer>\n.*?\n</answer>$"
        matches_pattern = [re.match(pattern, response, re.S) for response in responses]
        matches_pattern2 = [re.match(pattern2, response, re.S) for response in responses]
        format_rewards = []
        for match_pattern, match_pattern2 in zip(matches_pattern, matches_pattern2):
            if match_pattern:
                format_rewards.append(0.5)
            elif match_pattern2:
                format_rewards.append(0.5)
            else:
                format_rewards.append(0.0)
        rewards += torch.tensor(format_rewards, device=args.device)
        # 2. 标记奖励（防止严格奖励稀疏，仅针对训练推理模型时使用）
        def mark_num(text):
            reward = 0
            if text.count("<think>") == 1:  reward += 0.25
            if text.count("</think>") == 1:  reward += 0.25
            if text.count("<answer>") == 1:  reward += 0.25
            if text.count("</answer>") == 1:  reward += 0.25
            return reward
        mark_rewards = [mark_num(response) for response in responses]
        rewards += torch.tensor(mark_rewards, device=args.device)
        return rewards
    rewards = torch.zeros(len(responses), device=args.device)
```
##### 3. 完整的加权奖励
先看看这部分完整代码。
```python
# 使用reward model计算整个response的奖励
with torch.no_grad():
	reward_model_scores = []
	for prompt, response in zip(prompts, responses):
		pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
		matches = re.findall(pattern, prompt, re.DOTALL)
		messages = [{"role": role, "content": content.strip()} for role, content in matches]
		tmp_chat = messages + [{"role": "assistant", "content": response}]
		score = reward_model.get_score(reward_tokenizer, tmp_chat)
		scale = 3.0
		score = max(min(score, scale), -scale)
		# 当args.reasoning=1时，额外计算<answer>内容的奖励
		if args.reasoning == 1:
			answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
			if answer_match:
				answer_content = answer_match.group(1).strip()
				# 对answer内容单独计算reward
				tmp_chat = messages + [{"role": "assistant", "content": answer_content}]
				answer_score = reward_model.get_score(reward_tokenizer, tmp_chat)
				answer_score = max(min(answer_score, scale), -scale)
				score = score * 0.4 + answer_score * 0.6
		reward_model_scores.append(score)
	reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
	rewards += reward_model_scores
```
我们把整个回答(包括`<think>`和`<answer>`)喂给奖励模型(INternLM2-1.8B)，然后让它出一个全局评分，对应`tmp_chat = messages + [{"role": "assistant", "content": response}]; score = reward_model.get_score(reward_tokenizer, tmp_chat)`。这样就能判断整体语感、格式是否自然，是否有明显胡言乱语。
如果开了推理，那么我们还要再把思考`<think>`的内容隔离出来，重点看一下回答`<answer>`的内容。此时不管你推理模型思考的多好，裁判先生只看你答的好不好。对应`if answer_match:`这一部分。最后再把回答部分内容的分数和整体(思考+回答)的内容四六开算总分`score = score * 0.4 + answer_score * 0.6`，得到最终分数。
>那可不可以只给`<answer>`打分？如果只给回答打分，那PPO就退化成了SFT。这里保留40%是为了保护推理过程，确保思考的逻辑链条不崩塌。
##### 5. 损失
把损失单独拎出来分析一下，$L_{PPO}=-E[\min(r_t\cdot A_t,\operatorname{clip}(r_t,1-\varepsilon,1+\varepsilon)\cdot A_t)]+\beta\cdot E[\operatorname{KL}]$。策略项$f(r_t) = \min(r_t, \operatorname{clip}(r_t,1-\varepsilon,1+\varepsilon))$来控制裁剪概率比，防止更新过激；优势项$g(A_t)=R-V(s)$通过奖励模型估计价值函数；正则项$h(KL_t)=\beta\cdot E[\operatorname{KL}]$就是全局KL散度。
### GRPO(群体相对策略优化)
 对于同一个问题，让模型生成$N$个不同的回答，然后计算这$N$个回答的奖励分数。接着把这$N$个回答的平均奖励作为baseline，高于baseline的回答被鼓励，低于baseline被抑制。用这种方式可以巧妙地避免额外训练评论员Critic。
回忆一下，PPO中，由策略模型中的Actor生成多个回答后，用评论员Critic来给分。虽然Critic只是在后面接了一个`value_head`， 但是整个过程还是训练了两个网络。GRPO相当于给多个回答算平均分，然后用平均分来评定哪些该奖励，哪些该惩罚。此时Critic就可以离场了，重要的是裁判Reward的均值和标准差。
再来看看代码，数据这里和PPO完全一样，都是RLAIF的数据类型。区别在于训练方式和损失。
##### 奖励计算
和PPO一样，还是那个`calculate_rewards`函数，只不过细节有所变动。
```python
def calculate_rewards(prompts, responses, reward_model, reward_tokenizer):
    """整合所有奖励函数计算总奖励"""
	... # 前面推理阶段的评分逻辑完全不变
    with torch.no_grad():
        reward_model_scores = []
        batch_size = len(prompts)
        scale = 3.0
        for i in range(batch_size):
            for j in range(args.num_generations):
                response_idx = i * args.num_generations + j
                response = responses[response_idx]
                prompt = prompts[i]
                pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                matches = re.findall(pattern, prompt, re.DOTALL)
                messages = [{"role": role, "content": content.strip()} for role, content in matches]
                tmp_chat = messages + [{"role": "assistant", "content": response}]
                score = reward_model.get_score(reward_tokenizer, tmp_chat)
                score = max(min(score, scale), -scale)
                if args.reasoning == 1:
                    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
                    if answer_match:
                        answer_content = answer_match.group(1).strip()
                        tmp_chat = messages + [{"role": "assistant", "content": answer_content}]
                        answer_score = reward_model.get_score(reward_tokenizer, tmp_chat)
                        answer_score = max(min(answer_score, scale), -scale)
                        score = score * 0.4 + answer_score * 0.6
                reward_model_scores.append(score)
        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores
    return rewards
```
原本PPO只有一重循环`for prompt, response in zip(prompts, responses)`，每出一组提示词和回复，裁判都打一次分，一次backward只涉及`batch_size`条数据。但是GRPO有所不同，这里用了两重循环，分别是`for i in range(batch_size): \\ for j in range(args.num_generations)`，此时涉及的数据数量就是`batch_size * num_generations`，每个问题都会给多个答复。
##### 损失
GRPO 的损失函数最大的特点是去掉了Critic的预测值，直接利用组内的奖励分布来计算**优势(Advantage)**。那这个优势怎么算呢？对于同一个问题生成的$G$个回答，我们计算它们的奖励均值$\mu$和标准差$\sigma$。第$i$个回答的优势值为：$\hat{A}_i = \frac{r_i - \text{mean}(r_1, ..., r_G)}{\text{std}(r_1, ..., r_G)}$
有了优势值$\hat{A}_i$，剩下的逻辑和PPO一致，依然好似用Clip机制来保护模型。损失公式为：$L_{GRPO} = -E [r_t \cdot A_t - \beta \cdot KL_t]$。策略项为$\min(r_t\cdot A_t,\operatorname{clip}(r_t))$，用概率比来进行裁剪；优势项$g(A_t)=\frac{R-\mu _{group}}{\sigma _{group}}$做了组内归一化，消除Critic模型；正则项$h(KL_t)=\beta \cdot KL_t$就是在token级KL散度约束。
那么这里还会有个问题，*假设某个问题略难，导致N个回答的奖励分数几乎一样(大部分情况是一样烂而不是一样好)，那么这一组的学习信号就无限接近0。 在MiniMind这种超小模型上，这个问题尤为明显，求解数学问题99.99%的情况下整组回答质量都很差，那么将无法学习。 因此必须为模型指定合理的domain，即必须限制在能力边界内*。
## Reason
推理模型的训练有蒸馏和RL两种方式。面向数据进行蒸馏的方式没啥技术含量，相比与RL稍逊风骚。不过DeepSeek-R1实在太火了，它的论文指出`>3B`的模型经历多次反复的冷启动和RL奖励训练才能获得肉眼可见的推理能力提升。有必要简单了解一下。它的回复模板变成了
```json
<think>\n思考过程\n</think>\n
<answer>\n最终回答\n</answer>
```
大体训练思路其实和SFT差不多，代码可以直接运行，并且跑一下评估。


