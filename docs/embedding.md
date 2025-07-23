* embedding

```rust
// 参数设置
let vocab_size = 5;    // 词表有5个token (ID: 0,1,2,3,4)
let embedding_dim = 3;  // 每个token用3维向量表示

// 嵌入矩阵（随机初始化示例）
// 形状: (vocab_size, embedding_dim) = (5, 3)
let embedding_matrix = [
    [0.1, 0.2, 0.3],  // ID=0 的嵌入向量
    [0.4, 0.5, 0.6],  // ID=1
    [0.7, 0.8, 0.9],  // ID=2
    [1.0, 1.1, 1.2],  // ID=3
    [1.3, 1.4, 1.5],  // ID=4
];
```

* TokenID
```rust
let src_ids = [2, 0, 3];  // 序列长度为3，ID范围合法（均 < vocab_size=5）
```

* 查找过程
```rust
output = [
    embedding_matrix[2],  # 取第2行 -> [0.7, 0.8, 0.9]
    embedding_matrix[0],  # 取第0行 -> [0.1, 0.2, 0.3]
    embedding_matrix[3],  # 取第3行 -> [1.0, 1.1, 1.2]
]
```

