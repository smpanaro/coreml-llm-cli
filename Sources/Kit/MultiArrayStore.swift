import Foundation
import CoreML
import OSLog

/// Store and make accessible MLMultiArrays that will be reused
/// in either subsequent chunks or subsequent forward passes.
class MultiArrayStore {
    // Arrays that are used as inputs/outputs for multiple chunks.
    // Names must be unique across the whole pipeline.
    let sharedArrays: [String: MLMultiArray]
    // Arrays that are only used with one chunk.
    // Names must only be unique within a chunk. e.g. KV cache inputs/outputs.
    let chunkArrays: [[String: MLMultiArray]]

    // Map from output names to a different key name to use for the output
    // backing. Useful when the input is consumed and the output is the same shape.
    var outputBackingMapping: [String: String]

    let signposter = OSSignposter(subsystem: "com.stephenpanaro.llm-cli", category: "MultiArrayStore")

    init(sharedArrays: [String : MLMultiArray],
         chunkArrays: [[String : MLMultiArray]],
         outputBackingMapping: [String: String] = [:]
    ) {
        self.sharedArrays = sharedArrays
        self.chunkArrays = chunkArrays
        self.outputBackingMapping = outputBackingMapping
    }

    func inputFeatures(forChunk chunkIndex: Int, model: MLModel) -> [String: MLMultiArray] {
        Dictionary(uniqueKeysWithValues: model.modelDescription.inputDescriptionsByName.keys.compactMap { name in
            self.array(named: name, for: chunkIndex).map { arr in (name, arr) }
        })
    }

    func outputBackings(forChunk chunkIndex: Int, model: MLModel) -> [String: MLMultiArray] {
        Dictionary(uniqueKeysWithValues: model.modelDescription.outputDescriptionsByName.keys.compactMap { name in
            let backingName = outputBackingMapping[name] ?? name
            return self.array(named: backingName, for: chunkIndex).map { arr in (name, arr) }
        })
    }

    func update(outputs: MLFeatureProvider, forChunk chunkIndex: Int) {
        let state = signposter.beginInterval("Update Outputs", "Chunk \(chunkIndex)")
        defer { signposter.endInterval("Update Outputs", state) }

        outputs.featureNames.forEach { name in
            // Only float16 buffers are cached.
            guard let multiArrayValue = outputs.featureValue(for: name)?.multiArrayValue,
                  multiArrayValue.dataType == .float16
            else { return }

            guard let pixelBuffer = multiArrayValue.pixelBuffer
            else { preconditionFailure("output array is not pixel buffer backed: \(name)") }

            let cacheName = outputBackingMapping[name] ?? name
            guard let cachePixelBuffer = self.array(named: cacheName, for: chunkIndex)?.pixelBuffer
            else { preconditionFailure("output array is not present in the cache: \(name)") }

            // TODO: Gracefully degrade by manually copying the array.
            precondition(pixelBuffer === cachePixelBuffer, "output backing not used: \(name)")
        }
    }

    fileprivate func array(named name: String, for chunkIndex: Int) -> MLMultiArray? {
        if let sharedArr = sharedArrays[name] {
            return sharedArr
        } else if let chunkArr = chunkArrays[chunkIndex][name] {
            return chunkArr
        }
        return nil
    }
}

extension MultiArrayStore {
    convenience init(for pipeline: ModelPipeline) {
        var sharedArrays = [String: MLMultiArray]()
        var chunkArrays = [[String: MLMultiArray]]()

        // For each output key, reuse the input array named by the value.
        let outputBackingMapping = ["new_x": "x"]

        // KV caches, cos + sin for RoPE, attention mask
        pipeline.chunks.forEach { chunk in
            let model = chunk.model!
            var floatShapes = [String: [Int]]()

            let modelDescription = model.modelDescription
            let allDescriptions = modelDescription.inputDescriptionsByName.merging(
                modelDescription.outputDescriptionsByName, uniquingKeysWith: { a,b in a })

            allDescriptions.forEach { (name, desc) in
                guard let constraint = desc.multiArrayConstraint else { return }
                if constraint.dataType != .float16 { return }
                if outputBackingMapping.keys.contains(name) { return }
                floatShapes[name] = constraint.shape.map { $0.intValue }
            }

            let isShared = { (key: String) -> Bool in !key.contains("cache") }
            let sharedShapes = floatShapes.filter { isShared($0.key) }
            let chunkShapes = floatShapes.filter { !isShared($0.key) }

            // TODO: Sort so K caches are allocated close to other K caches. Ditto for V.
            //       See if it has a performance impact.

            sharedShapes.forEach { (name, shape) in
                if sharedArrays[name] == nil {
                    sharedArrays[name] = MLMultiArray.emptyIOSurfaceArray(shape: shape)!
                }
            }

            var currChunkArrays = [String:MLMultiArray]()
            chunkShapes.forEach { (name, shape) in
                currChunkArrays[name] = MLMultiArray.emptyIOSurfaceArray(shape: shape)!
            }
            chunkArrays.append(currChunkArrays)
        }

        self.init(sharedArrays: sharedArrays, chunkArrays: chunkArrays, outputBackingMapping: outputBackingMapping)
    }
}


extension MultiArrayStore {
    func featureProvider(forChunk chunkIndex: Int, model: MLModel, tokens: [Int]) throws -> MLFeatureProvider  {
        let state = signposter.beginInterval("Prepare Features", "Chunk \(chunkIndex)")
        defer { signposter.endInterval("Prepare Features", state) }

        var cacheFeatures = inputFeatures(forChunk: chunkIndex, model: model)

        // Input IDs is populated in concert with KV cache updates.
        //
        //                            0 1 2 3 4
        //  _ _ _ _ _ _ _ _ _ _ _ _ _
        // ┌─────────────────────────┬────────┐
        // └───────────Cache─────────┴─Inputs─┘
        //
        //
        //                            5 _ _ _ _
        //  _ _ _ _ _ _ _ _ 0 1 2 3 4
        // ┌─────────────────────────┬────────┐
        // └───────────Cache─────────┴─Inputs─┘
        //
        //
        //                            5 6 _ _ _
        //  _ _ _ _ _ _ _ _ 0 1 2 3 4
        // ┌─────────────────────────┬────────┐
        // └───────────Cache─────────┴─Inputs─┘
        //
        //            ┌ ─ ─ ─ ─ ─ ─ ┐
        //             Same for 7-9
        //            └ ─ ─ ─ ─ ─ ─ ┘
        //
        //                            5 6 7 8 9
        //  _ _ _ _ _ _ _ _ 0 1 2 3 4
        // ┌─────────────────────────┬────────┐
        // └───────────Cache─────────┴─Inputs─┘
        //
        //  ◀━━━━━━━━━━━━━Shift!━━━━━━━━━━━━━━━
        //
        //                             A _ _ _ _
        //   _ _ _ 0 1 2 3 4 5 6 7 8 9
        //  ┌─────────────────────────┬────────┐
        //  └───────────Cache─────────┴─Inputs─┘
        //
        //              ┌ ─ ─ ─ ─ ─
        //                 Repeat  │
        //              └ ─ ─ ─ ─ ─
        var padCount = 0
        let inputDescriptions = model.modelDescription.inputDescriptionsByName
        if let inputIDsConstraint = inputDescriptions["input_ids"]?.multiArrayConstraint {
            let inputShape = inputIDsConstraint.shape.map { $0.intValue }
            let inputLength = inputShape.last!

            // For inputLength 64, tokens.count -> suffixLength:
            // 0 -> 0, 5 -> 5, 64 -> 64, 65 -> 1, 128 -> 64
            let suffixLength = tokens.isEmpty ? 0 : (tokens.count - 1) % inputLength + 1
            let inputTokens = tokens.suffix(suffixLength).map { Int32($0) }
            padCount = max(0, inputLength - inputTokens.count)
            let paddedInputTokens = inputTokens + Array(repeating: Int32(0), count: padCount)
            let inputIDs = MLShapedArray<Int32>(scalars: paddedInputTokens, shape: inputShape)
            cacheFeatures["input_ids"] = MLMultiArray(inputIDs)
        }

        if inputDescriptions["full_sequence_length"]?.multiArrayConstraint != nil {
            // For example: [Cache0 Cache1][Input2 Input3 Pad0]
            // The causal mask prevents Pad0  from impacting
            // the InputN token predictions. We must include it
            // here so RoPE is correct for the InputN tokens.
            let fullSequenceLength = MLShapedArray(repeating: Int32(tokens.count + padCount), shape: [1])
            cacheFeatures["full_sequence_length"] = MLMultiArray(fullSequenceLength)
        }

        return try MLDictionaryFeatureProvider(dictionary: cacheFeatures)
    }
}
