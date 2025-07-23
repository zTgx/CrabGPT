```rust
let a = Tensor::new(&[
    [0f32, 1., 2.],
    [3.  , 4., 5.],
    [6.  , 7., 8.]
], &Device::Cpu).unwrap();

let b = a.narrow(0, 1, 2).unwrap(); 
```

* Debug -> `{:?}`
输出Tensor的 dim 信息。

* Display -> `{}`
输出具体的matrix 数据。