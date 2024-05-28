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

|Variant|First Load Time|Second+ Load Time|Tokens/Sec    |
|--     |--             |--               |--            |
|M1 Max |77s            |7.5s             |4.97 +/- 0.11 |

## Inference Optimizations
This CLI implements a couple optimizations that might be useful/interesting even in medium-sized and smaller models.

### CVPixelBuffer/IOSurface MLMultiArrays
All float16 model input and output arrays are created with IOSurface-backed CVPixelBuffers ([docs](https://developer.apple.com/documentation/coreml/mlmultiarray/3882834-init)). This avoids unnecessary copying that can occur when going between CPU and ANE, especially for large arrays like the KV cache.

### Model Chunking
The model is split into multiple smaller CoreML model chunks:
- 1 for the embedding+attention mask+RoPE cos/sin
- N for the blocks (3 [blocks](https://github.com/Lightning-AI/litgpt/blob/221b7ef54161272162aa9b036f1ef3674f3160a4/litgpt/model.py#L139) per chunk)
- 1 for the lm (lanaguage modeling) head

This allows for faster model loading and enables async KV cache manipulation (see below).

### Async KV Cache Updates
This model uses an ANE-compatible KV cache which needs to be shifted prior to each prediction. Due to the model chunking this does not need to happen synchronously in each chunk. It only needs to be updated before the next chunk prediction (~1 full forward pass in the future).

To take advantage of this, the new sections of the KV cache are returned from the model and a separate CoreML model combines them with the prior cache values asynchronously. For Llama this saves ~1-2ms per chunk (~20ms overall).

```
[ old KV cache (length 448) ]   [ hidden states (length 64) ]
                     ↘             ↙
                      [chunk model]
                     ↙             ↘
[ new KV cache (length 64)  ]   [ new hidden states (length 64) ]


Async after the chunk prediction completes:
[ old KV cache (length 448) ]   [ new KV cache (length 64) ]
                   ↘                     ↙
                    [ cache update model]
                              ↓
               [ updated KV cache (length 448) ]
```

This pattern would probably work for other non-critical path computations too.
