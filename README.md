# CoreML LLM CLI
CLI to demonstrate running a large language model (LLM) on Apple Neural Engine.

Download a CoreML-compatible Llama 2 7B model (~4GB), load it, and generate text:
```shell
$ swift run LLMCLI --repo-id smpanaro/Llama-2-7b-coreml
```

**Note:** macOS 14 (Sonoma) is required.

## Performance
I'm curious to see the performance of this model on different M-series chips. If you don't see your chip below or if you get significantly different timings, please run the following command and open an issue so I can add it!

To download + measure:
```shell
$ swift run -c release LLMCLI --repo-id smpanaro/Llama-2-7b-coreml --max-new-tokens 80
```

|Variant|1st Load Time|2nd+ Load Time|Tokens/Sec  |ANE Power|
|--     |--           |--            |--          |-        |
|M1 Max |113s         |8.1s          |7.02 ± 0.11 |4.2W     |

For M2-M4, consider the times for Model Version 1 as a lower bound:

<details><summary>Model Version 1 (a76a14d)</summary>

|Variant|1st Load Time|2nd+ Load Time|Tokens/Sec  |ANE Power|
|--     |--           |--            |--          |-        |
|M1 Max |77s          |7.5s          |4.97 ± 0.11 |3.6W     |
|M2     |-            |22.9s         |5.51 ± 0.56 |4.5-7.2W |
|M2 Pro |71s          |4.2s          |6.76 ± 0.09 |-        |
|M3     |64s          |5.47s         |7.12 ± 0.16 |5.6W     |
|M3 Max |-            |-             |7.6         |5.5W     |
|M4 iPad|66s          |-             |7.76 ± 0.36 |-        |

</details>

## Inference Optimizations
This CLI implements a couple optimizations that might be useful/interesting even in medium-sized and smaller models.

### CVPixelBuffer/IOSurface MLMultiArrays
All float16 model input and output arrays are created with IOSurface-backed CVPixelBuffers ([docs](https://developer.apple.com/documentation/coreml/mlmultiarray/3882834-init)). This avoids unnecessary copying that can occur when going between CPU and ANE, especially for large arrays like the KV cache.

### 20% Faster Convolutions With Reshaping
As recommended in Apple's [Deploying Transformers on the Apple Neural Engine](https://machinelearning.apple.com/research/neural-engine-transformers), this model uses a 4D tensor layout: (Batch, Channels, 1, Sequence). We process 64 tokens at a time, so most tensors are (B,C,1,64). It turns out that the convolutions in the MLP are 50% faster when the tensor is (B,C,8,8). This seems to hold across hardware versions (M1, A14, A16, A17 Pro) so we can leverage it for an extra speedup.

Unfortunately, attention requires a (B,C,1,S) shape tensor so we cannot simply run the whole model in (B,C,8,8) for the full 50% speedup. Instead we reshape to (B,C,1,64) before[^1] the QKV projections and back to (B,C,8,8) just before the attention out projection. This seems to minimize the cost of reshapes and allows us to achieve a ~20% overall speedup.

[^1]: Yes, before. Doing less reshapes (1 instead of 3) is faster than doing these smaller convolutions in (8,8).

### Model Chunking
The model is split into multiple smaller CoreML model chunks:
- 1 for the embedding+attention mask+RoPE cos/sin
- N for the blocks (3 [blocks](https://github.com/Lightning-AI/litgpt/blob/221b7ef54161272162aa9b036f1ef3674f3160a4/litgpt/model.py#L139) per chunk)
- 1 for the lm (lanaguage modeling) head

This allows for faster model loading and enables async KV cache manipulation (see below).

### Async KV Cache Updates
This model uses an ANE-compatible KV cache which needs to be shifted periodically. Due to the model chunking this does not need to happen synchronously in each chunk. It only needs to be updated before the next chunk prediction (~1 full forward pass in the future).

To take advantage of this, the new sections of the KV cache are returned from the model and a separate CoreML model combines them with the prior cache values asynchronously. For Llama this saves ~1-2ms per chunk (~20ms overall).

```
┌──────────────┐      ┌───────────────┐
│ Old KV Cache │      │ Hidden States │
│ (Length 448) │      │  (Length 64)  │
└──────────────┘      └───────────────┘
              ↘        ↙
             ┌───────────┐
             │Chunk Model│
             └───────────┘
              ↙        ↘
┌──────────────┐      ┌─────────────────┐
│ New KV Cache │      │New Hidden States│
│ (Length 64)  │      │   (Length 64)   │
└──────────────┘      └─────────────────┘

┌───────────────────────────────────────────┐
│Async after the chunk prediction completes:│
│                                           │
│  ┌──────────────┐     ┌──────────────┐    │
│  │ Old KV Cache │     │ New KV Cache │    │
│  │ (Length 448) │     │ (Length 64)  │    │
│  └──────────────┘     └──────────────┘    │
│               ↘         ↙                 │
│           ┌──────────────────┐            │
│           │Cache Update Model│            │
│           └──────────────────┘            │
│                    ↓                      │
│            ┌────────────────┐             │
│            │Updated KV Cache│             │
│            │  (Length 448)  │             │
│            └────────────────┘             │
└───────────────────────────────────────────┘
```

This pattern would probably work for other non-critical path computations too.
