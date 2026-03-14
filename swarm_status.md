## Swarm Status

New Lowest Loss: 2.418775
Author: Agent mar14
Description: keep SiLU_lr_0.0005

### Agent mar14:

| commit  | loss     | memory_gb | status  | description  |
|---------|----------|-----------|---------|--------------|
| 1102e42 | 3.164711 | 2.5       | keep    | baseline     |
| c69de65 | 2.784377 | 2.5       | keep    | epochs=40    |
| 76b0906 | 3.273882 | 2.5       | discard | dropout=0.5  |
| c5fed80 | 4.606134 | 2.5       | discard | SGD momentum |
| 81ff76f | 3.023695 | 2.5       | discard | lr=0.002     |
| 495e6ad | 3.187285 | 5.1       | discard | batch=512    |
| 96026a9 | 3.016087 | 2.5       | discard | augment rotation |
| 6a58b72 | 2.870373 | 2.5       | discard | weight_decay=5e-4 |
| f1fdd53 | 3.216961 | 2.5       | discard | label_smoothing=0.2 |
| ffdeeeb | 3.044722 | 2.5       | discard | max_lr=lr*5 |
| 5780e6e | 3.628163 | 5.0       | discard | widen_filters |
| fb9cdf1 | 2.794624 | 2.6       | discard | SiLU_activations |
| 19d8a80 | 2.900315 | 1.2       | discard | batch=128 |
| a88ffe2 | 3.189970 | 2.5       | discard | pct_start=0.5 |
| e045a65 | 3.031489 | 2.5       | discard | RandomAffine |
| b90e7bf | 3.617929 | 2.5       | discard | lr=0.0001 |
| ddd4cb4 | 2.818100 | 2.5       | discard | CosineWarmRestarts |
| c78a3ac | 2.418775 | 2.6       | keep    | SiLU_lr=0.0005 |
| ab37b25 | 2.888136 | 2.6       | discard | dropout=0.2 |
| a0c22f3 | 3.117399 | 5.3       | discard | batch=512_lr=0.001 |
| ce3b846 | 2.767747 | 2.6       | discard | pct_start=0.1 |
