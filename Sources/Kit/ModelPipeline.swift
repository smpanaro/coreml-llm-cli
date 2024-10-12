import Foundation
import CoreML
import OSLog

/// ModelPipeline manages the forward pass for an LLM that
/// has been split across many MLModels.
class ModelPipeline {
    let chunks: [PipelineChunk]
    var inferenceConfiguration: PipelineInferenceConfiguration? // Can't retrieve this until models are loaded.
    let cacheProcessorModel: DeferredModel
    let logitProcessor: LogitProcessor

    var loadableProcessors: [Loadable] {
        [cacheProcessorModel, logitProcessor]
    }

    let signposter = OSSignposter(subsystem: "com.stephenpanaro.llm-cli", category: "ModelPipeline")

    init(chunks: [PipelineChunk], cacheProcessor: DeferredModel, logitProcessor: LogitProcessor) {
        self.chunks = chunks
        precondition(chunks.count > 0)
        self.cacheProcessorModel = cacheProcessor
        self.logitProcessor = logitProcessor
    }

    /// Load the pipeline gradually to minimize resource usage
    /// during the initial load and model compilation/specialization.
    fileprivate func prewarm() {
        print("Compiling models: ", terminator: "")
        fflush(stdout)

        // No need for signposts, should be fast.
        loadableProcessors.forEach {
            $0.load()
            $0.unload()
        }

        chunks.enumerated().forEach { (i, chunk) in
            signposter.withIntervalSignpost("Prepare", "Warm Chunk \(i)") {
                chunk.load()
                chunk.unload()
            }
            print("*", terminator: "")
            fflush(stdout)
        }
        print()
    }

    func load() throws {
        prewarm()
        print("Loading models  : ", terminator: "")
        fflush(stdout)

        loadableProcessors.forEach {
            $0.load()
        }

        chunks.enumerated().forEach { (i, chunk) in
            signposter.withIntervalSignpost("Prepare", "Load Chunk \(i)") {
                chunk.load()
            }
            print("*", terminator: "")
            fflush(stdout)
        }
        print()

        inferenceConfiguration = .init(from: chunks.compactMap { $0.model })
        if inferenceConfiguration == nil {
            // Unable to infer the correct model parameters from the model inputs.
            // We won't be able to predict.
            throw PipelineError.unsupportedInferenceConfiguration
        }
    }

    func predict(tokens initialTokens: [Int], maxNewTokens: Int) throws -> AsyncThrowingStream<Prediction, Error> {
        guard let inferenceConfiguration else {
            throw PipelineError.unsupportedInferenceConfiguration
        }

        let arrayStore = MultiArrayStore(for: self)
        let kvCacheProcessor = KVCacheProcessor(pipeline: self, processorModel: cacheProcessorModel.model!)

        return AsyncThrowingStream<Prediction, Error> { continuation in
            let inputLength = inferenceConfiguration.inputLength
            var promptChunks = stride(from: 0, to: initialTokens.count, by: inputLength).map {
                Array(initialTokens[$0..<min($0 + inputLength, initialTokens.count)])
            }

            var tokens = promptChunks.removeFirst()
            let maxTokens = initialTokens.count + maxNewTokens

            let promptTimer = CodeTimer()
            var promptLatency: Measurement<UnitDuration>? = nil
            while tokens.count < maxTokens {
                let timer = CodeTimer()
                let tokenSignpostState = self.signposter.beginInterval("Predict", id: self.signposter.makeSignpostID(), "Token #\(tokens.count)")

                // Do a forward pass of the model.
                var logits: MLMultiArray!
                for (i, chunk) in self.chunks.enumerated() {
                    let model = chunk.model!

                    // Wait for the KV cache to be asynchronously updated.
                    try await kvCacheProcessor.wait(forChunk: i)

                    let inputs = try arrayStore.featureProvider(forChunk: i, model: model, tokens: tokens)

                    let options = MLPredictionOptions()
                    options.outputBackings = arrayStore.outputBackings(forChunk: i, model: model)

                    let predictState = self.signposter.beginInterval("Predict", id: self.signposter.makeSignpostID(), "Chunk \(i)")
                    let outputs = try await model.prediction(from: inputs, options: options)
                    self.signposter.endInterval("Predict", predictState)
                    arrayStore.update(outputs: outputs, forChunk: i) // Using output backings should make this ~a noop.

                    // Update the cache (async) each time a full set of input tokens is processed.
                    // e.g. if the model takes in 64 input tokens at once, update when tokens
                    // is length 64, 128, 192, etc.
                    if tokens.count % inputLength == 0 {
                        kvCacheProcessor.submit(inputs: inputs, outputs: outputs, forChunk: i)
                    }

                    if outputs.featureNames.contains("logits") {
                        logits = outputs.featureValue(for: "logits")!.multiArrayValue!
                    }
                }

                if !promptChunks.isEmpty {
                    for token in promptChunks.removeFirst() {
                        tokens.append(token)
                        continuation.yield(Prediction(newToken: token, allTokens: tokens, latency: nil, promptLatency: nil))
                    }
                    promptLatency = promptTimer.elapsed()
                } else {
                    let newTokenIndex = tokens.isEmpty ? 0 : (tokens.count - 1) % inputLength
                    let newToken = try await self.logitProcessor.argmax(logits: logits, index: newTokenIndex)
                    tokens.append(newToken)
                    continuation.yield(Prediction(newToken: newToken, allTokens: tokens, latency: timer.elapsed(), promptLatency: promptLatency))
                    promptLatency = nil
                }

                self.signposter.endInterval("Predict", tokenSignpostState, "\(tokens.last!)")
            }

            continuation.finish()
        }
    }
}


extension ModelPipeline: CustomDebugStringConvertible {
    var debugDescription: String {
        let fileName = chunks.first?.fileInfo.displayModelName ?? "<unknown>"
        return"\(Self.self) \(fileName) (\(chunks.count) chunks)"
    }
}

extension ModelPipeline {
    /// Creates a pipeline from the mlmodelc files in the given folder.
    /// Model files should follow the format: `${MODEL_PREFIX}_chunk${CHUNK_NUMBER}.mlmodelc`
    /// Does not load the model.
    class func from(
        folder: URL,
        modelPrefix: String?,
        cacheProcessorModelName: String,
        logitProcessorModelName: String,
        primaryCompute: MLComputeUnits = .cpuAndNeuralEngine,
        chunkLimit: Int? = nil
    ) throws -> ModelPipeline {
        let manager = FileManager.default
        let contents = try manager.contentsOfDirectory(atPath: folder.path(percentEncoded: false))

        let chunkFiles = contents
            .compactMap { ChunkFileInfo(url: folder.appending(path: $0)) }
            .filter { $0.url.pathExtension == "mlmodelc" }
            .filter {
                if let modelPrefix { $0.modelPrefix.hasPrefix(modelPrefix) }
                else { true }
            }
            .sorted(by: { $0.chunkNumber < $1.chunkNumber })

        let uniquePrefixes = Set(chunkFiles.map { $0.modelPrefix })
        if uniquePrefixes.count > 1 {
            throw PipelineError.ambiguousModelPath(possiblePrefixes: Array(uniquePrefixes))
        }

        let chunks = chunkFiles.enumerated()
            .filter { (i, _) in
                // Allow limiting the number of chunks for debugging.
                if i == 0 || i == chunkFiles.count - 1 { return true }
                if let chunkLimit { return i <= chunkLimit }
                return true
            }
            .map { (i, chunkFile) in
            let config = MLModelConfiguration()
            // The first chunk has operations that cannot run on ANE.
            config.computeUnits = i == 0 ? .cpuOnly : primaryCompute
            config.modelDisplayName = "Chunk \(chunkFile.chunkNumber)"
            return PipelineChunk(fileInfo: chunkFile, configuration: config)
        }

        if chunks.count == 0 {
            throw PipelineError.modelChunksNotFound
        }

        // Model for updating KV caches.
        let cacheProcessorURL = folder.appending(component: cacheProcessorModelName)
        let cacheModelConfig = MLModelConfiguration()
        cacheModelConfig.computeUnits = primaryCompute
        cacheModelConfig.modelDisplayName = "Cache Processor"
        let cacheProcessor = DeferredModel(url: cacheProcessorURL, configuration: cacheModelConfig)

        // Model for choosing the next token.
        let logitProcessorURL = folder.appending(component: logitProcessorModelName)
        let logitProcessorModelConfig = MLModelConfiguration()
        logitProcessorModelConfig.computeUnits = primaryCompute
        logitProcessorModelConfig.modelDisplayName = "Logit Processor"
        let logitProcessor = LogitProcessor(model: DeferredModel(url: logitProcessorURL, configuration: logitProcessorModelConfig))

        return ModelPipeline(chunks: chunks, cacheProcessor: cacheProcessor, logitProcessor: logitProcessor)
    }
}

class PipelineChunk {
    let fileInfo: ChunkFileInfo
    let configuration: MLModelConfiguration
    var model: MLModel?

    convenience init?(url: URL, configuration: MLModelConfiguration) {
        guard let fileInfo = ChunkFileInfo(url: url) else { return nil }
        self.init(fileInfo: fileInfo, configuration: configuration)
    }

    init(fileInfo: ChunkFileInfo, configuration: MLModelConfiguration) {
        self.fileInfo = fileInfo
        self.configuration = configuration
    }

    func load() {
        guard model == nil else { return }
        model = try! MLModel(contentsOf: fileInfo.url, configuration: configuration)
    }

    func unload() {
        model = nil
    }
}

extension PipelineChunk: CustomDebugStringConvertible {
    var debugDescription: String {
        return "\(Self.self) \(fileInfo.chunkNumber) (\(configuration.computeUnits.debugName))"
    }
}

class DeferredModel {
    let url: URL
    let configuration: MLModelConfiguration

    var model: MLModel?

    init(url: URL, configuration: MLModelConfiguration) {
        self.url = url
        self.configuration = configuration
    }

    func load() {
        model = try! MLModel(contentsOf: url, configuration: configuration)
    }

    func unload() {
        model = nil
    }
}

struct ChunkFileInfo {
    let url: URL
    let fileName: String
    let modelPrefix: String
    let chunkNumber: Int

    init?(url: URL) {
        self.url = url
        self.fileName = url.lastPathComponent

        var split = url.deletingPathExtension().lastPathComponent.split(separator: "_")
        guard
            let chunkString = split.popLast(),
            let chunkNumber = Int(chunkString.replacingOccurrences(of: "chunk", with: ""))
        else {
            return nil
        }

        self.modelPrefix = split.joined(separator: "_") + "_"
        self.chunkNumber = chunkNumber
    }

    var displayModelName: String {
        // Drop the last _
        String(modelPrefix.prefix(upTo: modelPrefix.index(before: modelPrefix.endIndex)))
    }
}

enum PipelineError: Error {
    case unsupportedInferenceConfiguration
    case cacheProcessorNotFound
    case modelChunksNotFound
    case ambiguousModelPath(possiblePrefixes: [String])
    case notImplementedError
}

struct Prediction {
    let newToken: Int
    let allTokens: [Int]
    let latency: Measurement<UnitDuration>?
    let promptLatency: Measurement<UnitDuration>?
}

struct CodeTimer {
    let start = CFAbsoluteTimeGetCurrent()

    func elapsed() -> Measurement<UnitDuration> {
        let seconds = CFAbsoluteTimeGetCurrent() - start
        return Measurement(value: seconds, unit: .seconds)
    }
}

protocol Loadable {
    func load()
    func unload()
}

extension LogitProcessor: Loadable {}
extension DeferredModel: Loadable {}
extension PipelineChunk: Loadable {}
