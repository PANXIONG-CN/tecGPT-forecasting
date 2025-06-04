# 计划：通过内存映射优化 DataLoader

**ID:** 1
**状态:** [ ] 未完成
**模块:** `src/utils/util.py`
**目标:** 重构 `DataLoader` 以使用内存映射（mmap）加载数据，防止将整个数据集加载到RAM中，实现按需加载批次。

## 1. 前期准备与分析

*   **[x] 审查用户提供的研究结果**：已详细阅读并理解用户关于 `DataLoader` 瓶颈、内存映射概念、`__init__` 和 `get_iterator._wrapper` 修改方案，以及 `util.load_dataset` 的调整需求。
*   **[ ] (可选) 审查 `src/utils/util.py` 的当前代码**：如果需要更深入的上下文，可以读取该文件。目前基于用户提供的信息，可以制定详细计划。

## 2. 详细实施步骤

### 操作 2.1: 修改 `util.load_dataset` 以传递内存映射数组

*   **[ ] 任务:** 在 `util.load_dataset` 函数中，为 `train.npz`、`val.npz` 和 `test.npz` 调用 `np.load` 时，添加 `mmap_mode='r'` 参数。
    *   **位置:** `src/utils/util.py` 内的 `load_dataset` 函数。
    *   **修改示例:**
        ```python
        # 原代码片段 (示意)
        # train_data_np = np.load(os.path.join(dataset_dir, "train.npz"))
        # val_data_np = np.load(os.path.join(dataset_dir, "val.npz"))
        # test_data_np = np.load(os.path.join(dataset_dir, "test.npz"))

        # 修改后代码片段 (示意)
        train_data_np = np.load(os.path.join(dataset_dir, "train.npz"), mmap_mode='r')
        val_data_np = np.load(os.path.join(dataset_dir, "val.npz"), mmap_mode='r')
        test_data_np = np.load(os.path.join(dataset_dir, "test.npz"), mmap_mode='r')
        ```
    *   **预期结果:** `train_data_np['x']`, `train_data_np['y']` (以及对应的val, test) 将成为内存映射对象，而不是完整的NumPy数组。

### 操作 2.2: 修改 `DataLoader.__init__` 以支持内存映射对象

*   **[ ] 任务:** 调整 `DataLoader` 的初始化逻辑，直接存储传入的内存映射对象，并处理相关的属性。
    *   **位置:** `src/utils/util.py` 内的 `DataLoader.__init__` 方法。
    *   **详细步骤:**
        1.  **[ ] 存储原始数据 (内存映射对象):**
            ```python
            # self.xs_orig = np.array(data_dict["x"]) # 旧代码
            # self.ys_orig = np.array(data_dict["y"]) # 旧代码
            self.xs_orig = data_dict["x"] # data_dict["x"] 现在是内存映射对象
            self.ys_orig = data_dict["y"] # data_dict["y"] 现在是内存映射对象
            ```
        2.  **[ ] 更新主数据引用:**
            ```python
            # self.xs = self.xs_orig.copy() # 旧代码
            # self.ys = self.ys_orig.copy() # 旧代码
            self.xs = self.xs_orig # 直接引用，不拷贝
            self.ys = self.ys_orig # 直接引用，不拷贝
            ```
        3.  **[ ] 调整 `size` 和 `original_size_before_padding`:** 确保这些属性正确反映内存映射对象的大小。对于内存映射对象，`shape` 属性应该仍然有效。
            ```python
            # self.size = self.xs.shape[0] # 应该保持不变
            # self.original_size_before_padding = data_dict["x"].shape[0] # 应该保持不变
            ```
        4.  **[ ] 调整填充逻辑 (`num_padding`):**
            *   **当前问题:** 不能直接向内存映射对象 `self.xs` 或 `self.ys` 追加数据（例如使用 `np.concatenate` 或 `np.vstack` 后再赋值）。
            *   **解决方案:**
                *   **[ ] 方案1 (首选):** 在索引层面处理填充。如果需要填充 `k` 个样本，并且最后一个有效样本的索引是 `N-1`，那么当请求的索引 `j >= N` 时，实际使用索引 `N-1` 对应的数据。这意味着 `get_iterator` 在获取数据时需要检查索引是否超出原始数据范围。
                *   **[ ] 方案2 (备选):** 单独加载最后一个真实样本到内存中 (`last_x_sample = np.array(self.xs_orig[-1:])`, `last_y_sample = np.array(self.ys_orig[-1:])`)。当需要填充数据时，从这个内存中的小数组重复获取。这需要在 `get_iterator` 中进行判断。
                *   **当前代码分析:** `DataLoader` 中似乎有 `self.xs = np.vstack([self.xs] + [self.xs[-1:, ...]] * num_padding)` 这样的逻辑。这需要修改。
                *   **修改方向:** 取消 `vstack` 操作。在 `get_iterator` 中，如果计算出的 `end_ind` 大于 `self.original_size_before_padding`，则对于超出部分，重复使用最后一个真实样本的数据。这意味着 `self.xs` 和 `self.ys` 将始终指向原始的、未填充的内存映射数据。`self.size` 属性将需要反映 *逻辑上* 填充后的大小，以便迭代器知道要生成多少批次。
            *   **[ ] 重新计算 `self.size`:**
                ```python
                self.original_size_unpadded = self.xs_orig.shape[0] # 原始未填充大小
                # self.size 在初始化时可能仍是 self.original_size_unpadded
                # 填充逻辑将在 get_iterator 或 shuffle_data 中通过索引间接处理
                # 或者，如果仍然需要一个表示填充后大小的 self.size:
                self.size = self.original_size_unpadded + self.num_padding # 总的逻辑大小
                ```

### 操作 2.3: 修改 `DataLoader.shuffle_data`

*   **[ ] 任务:** 改变数据打乱方式，从直接打乱数组变为打乱索引。
    *   **位置:** `src/utils/util.py` 内的 `DataLoader.shuffle_data` 方法。
    *   **详细步骤:**
        1.  **[ ] 生成并打乱索引:**
            ```python
            # 原代码可能直接打乱 self.xs, self.ys (如果它们是内存中的副本)
            # self.permutation = np.random.permutation(self.size) # 旧代码，self.size 可能是填充后的大小
            # self.xs = self.xs[self.permutation]
            # self.ys = self.ys[self.permutation]

            # 新代码:
            # 生成覆盖原始未填充数据大小的索引
            self.indices_unpadded = np.random.permutation(self.original_size_unpadded)
            # self.current_indices 将用于迭代，可能需要考虑填充
            # 一种方法是在迭代时处理填充索引，另一种是预先构造一个包含填充逻辑的完整索引列表
            # 为了简化，我们先打乱未填充部分的索引。填充部分的索引将在 get_iterator 中处理。
            ```
        2.  **[ ] 如何使用打乱的索引:** `get_iterator` 在取数据时，将使用 `self.indices_unpadded` 来访问 `self.xs` 和 `self.ys`。例如，第 `k` 个未填充样本的索引是 `self.indices_unpadded[k]`。
        3.  **修正与细化**
            ```python
            # 在 DataLoader.__init__ 中或 shuffle_data 首次调用时
            self.raw_indices = np.arange(self.original_size_unpadded)
            self.shuffled_indices = self.raw_indices.copy() # 初始化

            # 在 DataLoader.shuffle_data 方法中
            np.random.shuffle(self.shuffled_indices)
            # self.xs 和 self.ys (指向 mmap 对象) 保持不变
            ```

### 操作 2.4: 修改 `DataLoader.get_iterator._wrapper`

*   **[ ] 任务:** 按需从内存映射对象加载数据批次，并处理填充和索引。
    *   **位置:** `src/utils/util.py` 内的 `DataLoader.get_iterator` 方法返回的 `_wrapper` 内部函数。
    *   **详细步骤:**
        1.  **[ ] 迭代逻辑:** 循环将基于 `self.size` (逻辑上的总大小，包括填充)。
        2.  **[ ] 获取批次索引:**
            ```python
            # start_ind = i * self.batch_size
            # end_ind = min(start_ind + self.batch_size, self.size)
            ```
        3.  **[ ] 从内存映射对象中切片数据并转换为RAM中的NumPy数组:**
            *   对于每个样本索引 `j` in `range(start_ind, end_ind)`:
                *   计算实际的数据源索引 `source_idx`:
                    *   如果 `j < self.original_size_unpadded`:
                        *   `actual_data_idx = self.shuffled_indices[j]` (使用打乱后的索引从原始mmap数据中取)
                    *   如果 `j >= self.original_size_unpadded` (即在填充区域):
                        *   `actual_data_idx = self.shuffled_indices[self.original_size_unpadded - 1]` (使用打乱后最后一个真实样本的索引进行填充)
                           *  **或者更简单**：填充总是使用原始数据的最后一个样本 `self.original_size_unpadded - 1` (未经打乱的索引)，以避免打乱状态影响填充内容的一致性。这里需要明确选择一种策略。**倾向于使用原始数据的最后一个样本进行填充，即 `self.xs_orig[self.original_size_unpadded - 1]`**
                *   `x_sample = self.xs_orig[actual_data_idx, ...]`
                *   `y_sample = self.ys_orig[actual_data_idx, ...]`
            *   将收集到的 `x_sample` 和 `y_sample` 组成批次 `x_batch_mmap` 和 `y_batch_mmap`。
            *   **重要:** 将内存映射切片显式转换为新的NumPy数组:
                ```python
                # x_i = self.xs[start_ind:end_ind, ...] # 旧的直接切片
                # y_i = self.ys[start_ind:end_ind, ...]

                # 新的按索引逐个或小批量构建
                batch_x_list = []
                batch_y_list = []
                for k_logical_idx in range(start_ind, end_ind): # k_logical_idx 是从0到self.size-1的逻辑索引
                    if k_logical_idx < self.original_size_unpadded:
                        # 从真实数据获取
                        actual_physical_idx = self.shuffled_indices[k_logical_idx] # 使用打乱后的索引指向原始mmap数据
                        sample_x = self.xs_orig[actual_physical_idx]
                        sample_y = self.ys_orig[actual_physical_idx]
                    else:
                        # 从填充数据获取 (使用原始未打乱数据的最后一个物理样本)
                        sample_x = self.xs_orig[self.original_size_unpadded - 1]
                        sample_y = self.ys_orig[self.original_size_unpadded - 1]
                    
                    batch_x_list.append(np.array(sample_x)) # 确保转为内存中的 array
                    batch_y_list.append(np.array(sample_y))

                x_i = np.stack(batch_x_list)
                y_i = np.stack(batch_y_list)
                ```
        4.  **[ ] 转换为PyTorch张量:** 保持不变 (`torch.from_numpy(x_i).type(torch.FloatTensor)`).
        5.  **[ ] 确保内存映射文件句柄管理:** 通常，当内存映射对象（如 `self.xs_orig`）被垃圾回收时，文件句柄会自动关闭。只要 `DataLoader` 实例存在，这些对象就应该保持活动状态。如果 `DataLoader` 被删除，它们应该会被清理。

## 3. 潜在风险与应对

*   **[ ] 性能开销:** 频繁的小块磁盘读取可能比一次性大块读取慢。
    *   **缓解:** 批次大小（`batch_size`）不宜过小。可以测试不同批次大小对性能的影响。由于数据是按顺序或通过索引列表访问，操作系统和Numpy的mmap实现通常有预读（read-ahead）机制，可以缓解部分性能问题。
*   **[ ] 索引复杂性:** 处理原始索引、打乱索引和填充逻辑时，很容易出错。
    *   **缓解:** 编写清晰的单元测试来验证索引的正确性，以及数据批次是否按预期生成，特别是在边界条件（如数据集末尾、填充部分）。
*   **[ ] 文件句柄限制:** 如果同时打开大量内存映射文件（例如，每个数据块一个文件，且有很多数据块），可能会达到操作系统文件句柄的上限。
    *   **缓解:** 当前方案是针对 `train.npz`, `val.npz`, `test.npz` 三个主要文件，不太可能触发此问题。如果是更细粒度的文件分块，则需注意。
*   **[ ] `shuffle_data` 的影响:** 如果 `shuffle_data` 被多次调用（例如每个epoch都调用），确保索引的重新打乱是正确的，并且不会干扰正在进行的迭代（如果支持）。通常，`shuffle_data` 在每个epoch开始前调用一次。

## 4. 测试策略

*   **[ ] 单元测试 `DataLoader`:**
    *   **[ ] 测试初始化:** 验证 `self.xs_orig`, `self.ys_orig` 是内存映射对象，`self.size`, `self.original_size_unpadded` 等属性正确。
    *   **[ ] 测试数据批次获取 (不打乱):**
        *   验证获取的第一个批次、中间批次、最后一个真实数据批次的内容是否正确。
        *   验证涉及填充数据的批次内容是否正确（是否使用了最后一个真实样本）。
    *   **[ ] 测试数据批次获取 (打乱后):**
        *   验证打乱后，一个epoch内所有样本都被遍历一次且仅一次。
        *   验证打乱后，填充逻辑依然正确。
    *   **[ ] 测试边界条件:**
        *   数据集大小刚好是 `batch_size` 的整数倍。
        *   数据集大小不是 `batch_size` 的整数倍，需要填充。
        *   数据集很小（例如只有几个样本）。
*   **[ ] 集成测试:**
    *   **[ ] 使用修改后的 `DataLoader` 运行一个简化的训练流程** (例如，只跑一个或几个epoch，使用一个小型的虚拟 `.npz` 数据集，该数据集可以手动创建并包含可识别的模式)。
    *   **[ ] 监控内存使用情况:** 确认内存使用显著低于旧的加载方式。

## 5. Mermaid 图表 (示意 DataLoader 核心流程)

```mermaid
graph TD
    A[调用 load_dataset mmap_mode='r'] --> B(返回内存映射对象<br>np_mmap_x, np_mmap_y);
    B --> C[DataLoader.__init__];
    C --> D{存储 self.xs_orig = np_mmap_x<br>self.ys_orig = np_mmap_y};
    D --> E[计算 self.original_size_unpadded<br>self.size (逻辑大小)<br>self.shuffled_indices = np.arange(original_size_unpadded)];
    
    F[调用 shuffle_data (可选,每个epoch初)] --> G[np.random.shuffle(self.shuffled_indices)<br>打乱 self.shuffled_indices];
    
    H[调用 get_iterator] --> I{循环获取批次 i = 0..num_batches-1};
    I --> J[计算逻辑上的 start_idx, end_idx for batch i];
    J --> K{For k_logical_idx in start_idx..end_idx-1};
    K -- 真实数据? --> L{k_logical_idx < self.original_size_unpadded?};
    L -- Yes --> M[actual_physical_idx = self.shuffled_indices[k_logical_idx]];
    M --> N[sample_x = self.xs_orig[actual_physical_idx]<br>style N fill:#ccffcc];  % Green for processing
    N --> O[sample_y = self.ys_orig[actual_physical_idx]<br>style O fill:#ccffcc];
    L -- No (填充) --> P[sample_x = self.xs_orig[self.original_size_unpadded - 1]<br>style P fill:#ccffcc]; % Use last raw physical sample
    P --> Q[sample_y = self.ys_orig[self.original_size_unpadded - 1]<br>style Q fill:#ccffcc];
    O --> R[batch_x_list.append(np.array(sample_x))<br>style R fill:#lightblue]; % Blue for input/staging
    Q --> R;
    R -- batch_y_list.append --> S[batch_y_list.append(np.array(sample_y))<br>style S fill:#lightblue];
    S --> T{End For k_logical_idx};
    T --> U[x_batch = np.stack(batch_x_list)<br>style U fill:#orange]; % Orange for output
    U --> V[y_batch = np.stack(batch_y_list)<br>style V fill:#orange];
    V --> W[torch.from_numpy(x_batch)];
    W --> X[yield x_tensor, y_tensor];
    X --> I;
```

## 6. 清理与后续

*   **[ ] 移除旧的、不必要的代码或注释。**
*   **[ ] 添加清晰的中文注释解释新的内存映射逻辑、索引处理和填充策略。**

**风险分析与应对措施总结：**

1.  **索引管理错误**：
    *   **风险**：在处理原始大小、填充、打乱索引时，很容易引入 off-by-one 错误或不一致。
    *   **应对**：进行彻底的单元测试，覆盖各种边界情况。在实现时，对索引的计算和使用进行多次核对。使用清晰的变量名来区分不同类型的索引（例如 `logical_batch_idx`, `unpadded_data_idx`, `shuffled_raw_idx`）。
2.  **性能问题**：
    *   **风险**：虽然mmap避免了RAM溢出，但频繁的小I/O操作如果处理不当（例如，在内部循环中对mmap对象进行非常小的、非连续的切片），可能会比一次性加载到RAM中然后切片要慢。
    *   **应对**：确保在`get_iterator`中，当从mmap对象提取数据时，尽可能按批次或较大的块进行读取。Numpy的mmap切片通常是高效的。测试实际的读取性能，如果成为瓶颈，考虑是否可以在`get_iterator`中预先读取一个稍大的"超级块"到内存，然后从中划分小批次。但此举会增加RAM消耗，需权衡。
3.  **填充逻辑的复杂性**：
    *   **风险**：确保填充的样本正确（通常是最后一个有效样本），并且在打乱数据时，填充逻辑仍然一致。
    *   **应对**：明确填充策略——是使用原始数据集的最后一个样本，还是打乱后的逻辑序列中的最后一个有效样本。前者实现更简单。在测试中验证填充数据的正确性。
4.  **与现有代码的兼容性**：
    *   **风险**：`DataLoader`的接口（如`get_iterator`返回的内容）需要保持不变，以免破坏上层训练循环的逻辑。
    *   **应对**：严格遵守现有接口。主要变化在于内部实现。输出的批次 `x_i, y_i` 仍应是标准的PyTorch张量。

这个计划提供了详细的步骤和考虑因素。在实际编码之前，建议先仔细阅读 `src/utils/util.py` 中 `DataLoader` 的现有实现，以便更精确地定位修改点。 