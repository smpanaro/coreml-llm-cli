import Foundation
import CoreML
import OSLog

/// Asynchronously use a chunk's KV cache output to prepare
/// its KV cache inputs for the next predicition.
class KVCacheProcessor {
    var chunkTasks: [Task<Int, Error>?]
    let processorModel: MLModel

    let signposter = OSSignposter(subsystem: "com.stephenpanaro.llm-cli", category: "KVCacheProcessor")

    init(chunkCount: Int, processorModel: MLModel) {
        self.chunkTasks = Array(repeating: nil, count: chunkCount)
        self.processorModel = processorModel
    }

    convenience init(pipeline: ModelPipeline, processorModel: MLModel) {
        self.init(chunkCount: pipeline.chunks.count, processorModel: processorModel)
    }

    func submit(inputs: MLFeatureProvider, outputs: MLFeatureProvider, forChunk chunkIndex: Int) {
        let pair = ChunkInputsOutputs(inputs: inputs, outputs: outputs)
        chunkTasks[chunkIndex] = Task<Int, Error> {
            try await withThrowingTaskGroup(of: Int.self) { group in
                for blockIndex in 0..<inputs.blockCount() {
                    group.addTask {
                        try await self.predict(pair: pair, chunkIndex: chunkIndex, blockIndex: blockIndex)
                        return 0
                    }
                }
                try await group.waitForAll()
                return 0
            }
        }
    }

    fileprivate func predict(pair: ChunkInputsOutputs, chunkIndex: Int, blockIndex: Int) async throws {
        let state = signposter.beginInterval("Predict",
                                             id: signposter.makeSignpostID(),
                                             "Chunk \(chunkIndex) Block \(blockIndex)")
        defer { signposter.endInterval("Predict", state) }

        let inputs = try pair.cacheProcessorInputs(forBlock: blockIndex)
        let opts = pair.cacheProcessorOptions(forBlock: blockIndex)
        let outputs = try await self.processorModel.prediction(from: inputs, options: opts)

        let ignoredBackings = opts.ignoredOutputBackingKeys(outputs)
        assert(ignoredBackings.isEmpty, "Output backings were ignored: \(ignoredBackings)")
    }

    /// Block until any pending task for the given chunk is complete.
    func wait(forChunk chunkIndex: Int) async throws {
        guard let task = chunkTasks[chunkIndex]
        else { return }

        let state = signposter.beginInterval("Wait", "Chunk \(chunkIndex)")
        let _ = try await task.value
        signposter.endInterval("Wait", state)
    }
}

extension MLFeatureProvider {
    func blockCount() -> Int {
        let maxIndex = featureNames.filter {
            $0.contains("k_cache")
        }.compactMap {
            $0.components(separatedBy: "_").last
        }.compactMap {
            Int($0)
        }.max()
        guard let maxIndex else { return 0 }
        return maxIndex + 1
    }
}

struct ChunkInputsOutputs {
    let inputs: MLFeatureProvider
    let outputs: MLFeatureProvider

    func cacheProcessorInputs(forBlock blockIndex: Int) throws -> MLFeatureProvider {
        let inputs = [
            "old_k_cache": inputs.featureValue(for: "k_cache_\(blockIndex)")!.multiArrayValue!,
            "new_k_cache": outputs.featureValue(for: "new_k_cache_\(blockIndex)")!.multiArrayValue!,
            "old_v_cache": inputs.featureValue(for: "v_cache_\(blockIndex)")!.multiArrayValue!,
            "new_v_cache": outputs.featureValue(for: "new_v_cache_\(blockIndex)")!.multiArrayValue!,
        ]
        return try MLDictionaryFeatureProvider(dictionary: inputs)
    }

    func cacheProcessorOptions(forBlock blockIndex: Int) -> MLPredictionOptions {
        let opts = MLPredictionOptions()
        opts.outputBackings = [
            "updated_k_cache": inputs.featureValue(for: "k_cache_\(blockIndex)")!.multiArrayValue!,
            "updated_v_cache": inputs.featureValue(for: "v_cache_\(blockIndex)")!.multiArrayValue!,
        ]
        return opts
    }
}
