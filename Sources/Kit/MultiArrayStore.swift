import Foundation
import CoreML
import OSLog
import Accelerate

/// Store and make accessible MLMultiArrays that will be reused
/// in either subsequent chunks or subsequent forward passes.
class MultiArrayStore {
    // Arrays that are used as inputs/outputs for multiple chunks.
    // Names must be unique across the whole pipeline.
    fileprivate var sharedArrays: [String: MLMultiArray]
    // Arrays that are only used with one chunk.
    // Names must only be unique within a chunk. e.g. KV cache inputs/outputs.
    fileprivate var chunkArrays: [[String: MLMultiArray]]

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
            // Always initialize to the prompt model.
            let model = chunk.promptModel!
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

    func resize(for pipeline: ModelPipeline, with configuration: PipelineInferenceConfiguration) async {
        for (index, chunk) in pipeline.chunks.enumerated() {
            let model = chunk.model(forTokenCount: configuration.contextLength, newTokenCount: configuration.inputLength)

            let modelDescription = model.modelDescription
            let allDescriptions = modelDescription.inputDescriptionsByName.merging(
                modelDescription.outputDescriptionsByName, uniquingKeysWith: { a,b in a })

            let inputCacheKeys = modelDescription.inputDescriptionsByName.keys.filter {
                $0.contains("cache")
            }

            // Cache I/O require special handling.
            for inputCacheKey in inputCacheKeys {
                let outputCacheKey = "new_\(inputCacheKey)"
                assert(modelDescription.outputDescriptionsByName.keys.contains(outputCacheKey), "\(outputCacheKey) not found")

                let inputArray = chunkArrays[index][inputCacheKey]
                let newInputShape = modelDescription.inputDescriptionsByName[inputCacheKey]!.multiArrayConstraint!.shape.map { $0.intValue }
                let outputArray = chunkArrays[index][outputCacheKey]!
                let newOutputShape = modelDescription.outputDescriptionsByName[outputCacheKey]!.multiArrayConstraint!.shape.map { $0.intValue }

                let (newInputArray, _) = await self.resizedCacheArrays(inputArray: inputArray, outputArray: outputArray, newInputShape: newInputShape, newOutputShape: newOutputShape)

                chunkArrays[index][inputCacheKey] = newInputArray
//                chunkArrays[index][outputCacheKey] = newOutputArray // Unused for static sized cache.
            }

            for (name, desc) in allDescriptions {
                guard let constraint = desc.multiArrayConstraint else { continue }
                if constraint.dataType != .float16 { continue }
                if outputBackingMapping.keys.contains(name) { continue }

                // Cache handled above.
                if name.contains("cache") { continue }

                let newShape = constraint.shape.map { $0.intValue }
                if let arr = sharedArrays[name] {
                    sharedArrays[name] = await resizedArray(arr, name: name, toShape: newShape)
                }
                else if let arr = chunkArrays[index][name] {
                    chunkArrays[index][name] = await resizedArray(arr, name: name, toShape: newShape)
                }
                else {
                    print("missed \(name)")
                }
            }

            // TODO: Conditionalize this somehow.
            outputBackingMapping.merge([
                "new_k_cache_0": "k_cache_0",
                "new_k_cache_1": "k_cache_1",
                "new_k_cache_2": "k_cache_2",
                "new_v_cache_0": "v_cache_0",
                "new_v_cache_1": "v_cache_1",
                "new_v_cache_2": "v_cache_2"], uniquingKeysWith: {a,b in a})
        }
    }

    fileprivate func resizedCacheArrays(
        inputArray: MLMultiArray?,
        outputArray: MLMultiArray,
        newInputShape: [Int],
        newOutputShape: [Int]
    ) async -> (MLMultiArray, MLMultiArray) {
        guard #available(macOS 15, *) else {
            fatalError("multi-function models not supported")
        }

        if let inputArray, inputArray.isShape(newInputShape) && outputArray.isShape(newOutputShape) {
            return (inputArray, outputArray)
        }

        // We only support the transition from prompt -> generation (aka small/no input cache -> big input cache).
        assert(inputArray == nil || inputArray!.shape.last!.intValue < newInputShape.last!, "Can only increase the input cache sizes.")

        // Shortcut path for the case where the prompt output is exactly
        // the same shape as the generation input.
        if outputArray.isShape(newInputShape) {
            let newInputCacheArray = outputArray
            let newOutputCacheArray = MLMultiArray.emptyIOSurfaceArray(shape: newOutputShape)!
            return (newInputCacheArray, newOutputCacheArray)
        }

        // For reference, but the following should be unused.

        // e.g. old: input(448), output(64)
        //      new: input(511),  output(1)
        // At this point, output is the full output from the last prediction,
        // input is the input cache that produced that output.

        // Concatenate caches along the last axis.
        let oldOutput = outputArray.toTensor()
        let oldInput = inputArray!.toTensor()
        let fullCache = oldInput.concatenated(with: oldOutput, alongAxis: -1)
        let newStart = fullCache.shape.last! - newInputShape.last!
        let newInputCache = fullCache[..., newStart...]
        var newInputCacheShapedArray = await newInputCache.shapedArray(of: Float16.self)

        let newInputCacheArray = MLMultiArray.emptyIOSurfaceArray(shape: newInputShape)!
        let newOutputCacheArray = MLMultiArray.emptyIOSurfaceArray(shape: newOutputShape)!

        // We need to manually copy here because if we convert the shaped array directly to MLMultiArray
        // it is not IOSurface-backed.

        newInputCacheShapedArray.withUnsafeMutableShapedBufferPointer { ptr, shape, strides in
            guard let pixelBuffer = newInputCacheArray.pixelBuffer else { fatalError("pixel buffer is required") }

            CVPixelBufferLockBaseAddress(pixelBuffer, .init())
            defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .init()) }

            let height = CVPixelBufferGetHeight(pixelBuffer)
            let width = CVPixelBufferGetWidth(pixelBuffer)
            let rowBytes = CVPixelBufferGetBytesPerRow(pixelBuffer)

            // Dangerously assume the layout is the same.
            var src = vImage_Buffer(data: ptr.baseAddress,
                                    height: vImagePixelCount(height),
                                    width: vImagePixelCount(width),
                                    rowBytes: width * 2) // 2 bytes for float16
            var dest = vImage_Buffer(data: CVPixelBufferGetBaseAddress(pixelBuffer),
                                     height: vImagePixelCount(height),
                                     width: vImagePixelCount(width),
                                     rowBytes: rowBytes)

            let res = vImageCopyBuffer(&src, &dest, 2, vImage_Flags())
            assert(res == vImage_Error(kvImageNoError))
        }

//        let destArray = MLShapedArray<Float16>(newInputCacheArray)
//        let srcScalars = newInputCacheShapedArray.scalars
//        let destScalars = destArray.scalars
//        assert(newInputCacheShapedArray.scalarCount == destArray.scalarCount)
//        for i in 0..<newInputCacheShapedArray.scalarCount {
//            assert(srcScalars[i] == destScalars[i], "i \(i)")
//        }

        return (newInputCacheArray, newOutputCacheArray)
    }

    fileprivate func resizedArray(_ fromArray: MLMultiArray, name: String, toShape: [Int]) async -> MLMultiArray {
        if fromArray.isShape(toShape) {
            return fromArray
        }
        return MLMultiArray.emptyIOSurfaceArray(shape: toShape)!
    }
}


extension MultiArrayStore {
    func featureProvider(forChunk chunkIndex: Int, model: MLModel, tokens: [Int]) async throws -> MLFeatureProvider  {
        let state = signposter.beginInterval("Prepare Features", "Chunk \(chunkIndex)")
        defer { signposter.endInterval("Prepare Features", state) }

        var cacheFeatures = inputFeatures(forChunk: chunkIndex, model: model)

        let inputDescriptions = model.modelDescription.inputDescriptionsByName
        if let inputIDsConstraint = inputDescriptions["input_ids"]?.multiArrayConstraint {
            let shape = inputIDsConstraint.shape.map { $0.intValue }
            let padCount = max(0, shape.last! - tokens.count)
            let paddedTokens = Array(repeating: Int32(0), count: padCount) + tokens.suffix(shape.last!).map { Int32($0) }
            let inputIDs = MLShapedArray<Int32>(scalars: paddedTokens, shape: shape)
            cacheFeatures["input_ids"] = MLMultiArray(inputIDs)
        }

        if inputDescriptions["full_sequence_length"]?.multiArrayConstraint != nil {
            let fullSequenceLength = MLShapedArray(repeating: Int32(tokens.count), shape: [1])
            cacheFeatures["full_sequence_length"] = MLMultiArray(fullSequenceLength)
        }

        return try MLDictionaryFeatureProvider(dictionary: cacheFeatures)
    }
}

extension MLMultiArray {
    func isShape(_ shape: [Int]) -> Bool {
        let selfShape = self.shape.map { $0.intValue }
        if selfShape.count != shape.count {
            return false
        }
        return zip(selfShape, shape).allSatisfy({ $0 == $1 })
    }
}
