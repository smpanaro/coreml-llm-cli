# CoreML LLM CLI
CLI to demonstrate running a large language model (LLM) on Apple Neural Engine.

Download a CoreML-compatible Llama 2 7B model (~4GB), load it, and generate text:
```shell
$ swift run LLMCLI --repo-id smpanaro/Llama-2-7b-coreml --repo-directory sequoia
```

**Note:** macOS 15 (Sequoia) is required for this branch.

## Performance
I'm curious to see the performance of this model on different M-series chips. If you don't see your chip below or if you get significantly different timings, please run the following command and open an issue so I can add it!

To download + measure:
```shell
$ swift run -c release LLMCLI --repo-id smpanaro/Llama-2-7b-coreml --repo-directory sequoia --max-new-tokens 80
```

macOS 14 (Sonoma)
|Variant|First Load Time|Second+ Load Time|Tokens/Sec  |ANE Power|
|--     |--             |--               |--          |-        |
|M1 Max |77s            |7.5s             |4.97 ± 0.11 |3.6W     |
|M2     |-              |22.9s            |5.51 ± 0.56 |4.5-7.2W |
|M3     |64s            |5.47s            |7.12 ± 0.16 |5.6W     |
|M3 Max |-              |-                |7.6         |5.5W     |
|M4 iPad|66s            |-                |7.76 ± 0.36 |-        |

macOS 15 (Sequoia beta 1)
|Variant|First Load Time|Second+ Load Time|Prompt Processing|Tokens/Sec  |ANE Power|
|--     |--             |--               |--               |--          |-        |
|M1 Max |161s           |0.72s            |1.053s           |5.58 ± 0.35 |1.4W     |

macOS 15 (Sequoia beta 3)
|Variant|First Load Time|Second+ Load Time|Prompt Processing|Tokens/Sec   |ANE Power|
|--     |--             |--               |--               |--           |-        |
|M1     |~90s           |2.91s            |1.15s            |4.26 ± 0.64  |-        |
|M3 Pro |~70s           |0.5s             |0.73s            |11.72 ± 0.29 |-        |

macOS 15 (Sequoia beta 4, includes cache optimizations)
|Variant|First Load Time|Second+ Load Time|Prompt Processing|Tokens/Sec  |ANE Power|
|--     |--             |--               |--               |--          |-        |
|M1 Max |158s           |0.74s            |1.04s            |6.87 ± 0.13 |1.4W     |

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

### Async KV Cache Updates (macOS 14 Only)
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

## Additional macOS 15 (Sequoia) Optimizations/Workarounds
This branch has experimental support for the macOS 15. A few changes and workarounds were used to get it running.

### Multi-function Model Support
This repo leverages the new multi-function model support to ship two models with the same weights:
- A cache pre-fill model that takes inputs up to the full context length.
- A generation model that takes a single input.

This makes the initial model load slower (~2x) but the trade-off is that after that token generation is faster and power usage is lower (since the amount of work is much less). Plus, cached model loads are very fast on Sequoia.

Currently there is only a single generation model function that takes 1 input and uses a 512 token context. It's possible that adding more functions to the model with smaller context sizes could speed up generation at the cost of a slower initial load (e.g. use the 64 context size, then switch to the 128 context size etc). It might not help though depending on what the latency bottleneck is.

### RMSNorm Overflows
Unfortunately, the common RMSNorm implementation (e.g. from lit-gpt) overflows float16 for certain tokens (e.g. `<bos>`) and this catastrophically degrades the model when there is only a single input token.

Instead we use an alternative implementation based on the `reduce_l2_norm` CoreML operation which is less prone to overflowing. Roughly:

```python
B,C,_,S = x.shape
eps = torch.ones(B,1,1,S) * (1e-5 * x.size(1)) ** 0.5
xeps = torch.cat([x, eps], dim=1)
scale = torch.linalg.norm(xeps, dim=1, keepdim=True)
x = x / xeps
```

This works both on Sonoma as well as Sequoia beta 3+.

### Sync KV Cache Updates
Since we have the luxury of separate prompt and generation models and a static context size, we can update the KV cache appropriately without using a secondary model (as in "Async KV Cache Updates" above). This incurs a slight overhead in each model chunk (possibly negligible), but it is simpler:

- There is no need for an input cache in the prompt model.
- The input cache for generation model is the same shape as the output caches for both models.
    - By using the input arrays as the output backings for the output arrays, we never have to manually copy the KV cache!
- The output cache is always computed from the full cache as: `full_cache[...,1:]`

Additionally, the KV cache update model increases the CPU usage during generation from ~13% to ~60% on my M1 Max. Skipping it avoids the unnecessary excess. (From Instruments it looks like this might be caused by some copying/casting, will file a feedback if this persists in beta 3.)

### Model Load Sequencing
One downside of separate prompt and generation models is that loading and retaining both at the same time incurs overhead that is otherwise absent when retaining just one of them. It's possible that this is limited to the M1 series but it is significant: ~5-6 tok/sec → ~3-4 tok/sec.

The overhead can be seen in Instruments[^1] as calls to `H11ANEIn::ProgramReMap(H11ANEProgramBufferParamsStruct*, bool)` that occur at the start of a request to the ANE. When both the prompt and generation MLModel instances are loaded and retained, this overhead occurs between most, if not all, chunks of the generation pipeline and adds 7-10ms each.

To avoid this, we load the prompt model chunks first and process the prompt. Then we unload all of those MLModels (set them equal to nil) and afterwards load each chunk of the generation model. This does not increase the total load processing time, just rearranges when it occurs.

```
                                        ┌────────────────────────┐
┌─────────────────────────┐┌──────────┐┌┤Slow (Lots of Overhead) ├
│     Load Prompt and     ││ Process  ││└────────────────────────┘
│    Generation Chunks    ││  Prompt  ││    Generate Tokens...
└─────────────────────────┘└──────────┘└──────────────────────────

                                        ┌───────────────────────┐
┌────────────┐┌──────────┐ ┌──────────┐┌┤ Fast (Lower Overhead) ├─
│Load Prompt ││ Process  │││Load Gen. ││└───────────────────────┘
│   Chunks   ││  Prompt  │││  Chunks  ││    Generate Tokens...
└────────────┘└──────────┘│└──────────┘└──────────────────────────
                          ▼
                    Unload Prompt
                       Chunks
```

[^1]: Check both "High Frequency" and "Record Kernel Callstacks" for the Time Profiler

### Input:Cache Ratio
This model has a 512 context window. After prompt processing the two primary inputs to the model are:

- `input_ids`: the newest N tokens
- `kv_cache`: the KV cache for the oldest 512-N tokens

Interestingly N = 4 is marginally (~3%) but consistently faster than N = 1 so we use that.
