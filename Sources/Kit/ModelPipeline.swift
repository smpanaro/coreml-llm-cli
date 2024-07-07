import Foundation
import CoreML
import OSLog

/// ModelPipeline manages the forward pass for an LLM that
/// has been split across many MLModels.
class ModelPipeline {
    let chunks: [PipelineChunk]
    var inferenceConfiguration: PipelineInferenceConfiguration? // Can't retrieve this until models are loaded.
    let cacheProcessorModel: DeferredModel
//    let promptCacheProcessorModel: DeferredModel // TODO
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
    fileprivate func prewarm() async {
        print("Compiling models: ", terminator: "")
        fflush(stdout)

        // No need for signposts, should be fast.
        for processor in loadableProcessors {
            await processor.load()
            processor.unload()
        }

        for (i, chunk) in chunks.enumerated() {
            let state = signposter.beginInterval("Prepare", id: signposter.makeSignpostID(), "Warm Chunk \(i)")
            await chunk.load()
            chunk.unload()
            signposter.endInterval("", state)
            print("*", terminator: "")
            fflush(stdout)
        }
        print()
    }

    func load() async throws {
//        await prewarm() // Doesn't seem to help on Sequoia since model caching is iffy (in beta 1 at least).
        print("Loading models  : ", terminator: "")
        fflush(stdout)

        // Should be fast, sync is fine.
        for processor in loadableProcessors {
             await processor.load()
        }

        for (i, chunk) in chunks.enumerated() {
            let state = signposter.beginInterval("Prepare", id: signposter.makeSignpostID(), "Load Chunk \(i)")
            await chunk.load()
            signposter.endInterval("", state)
            print("*", terminator: "")
            fflush(stdout)
        }
        print()

        inferenceConfiguration = .init(from: chunks.compactMap { $0.promptModel! })
        if inferenceConfiguration == nil {
            // Unable to infer the correct model parameters from the model inputs.
            // We won't be able to predict.
            throw PipelineError.unsupportedInferenceConfiguration
        }
    }

    func predict(tokens: [Int], maxNewTokens: Int) throws -> AsyncThrowingStream<Prediction, Error> {
        guard let inferenceConfiguration else {
            throw PipelineError.unsupportedInferenceConfiguration
        }
        guard tokens.count <= inferenceConfiguration.inputLength else {
            // TODO: Support long prompts with a prompt-cache-processor model
            //       that shifts the KV cache by 64 each time to process the prompt
            //       `inputLength` tokens at a time.
            throw PipelineError.unimplementedLongPrompt
        }

        let arrayStore = MultiArrayStore(for: self)
        let kvCacheProcessor = KVCacheProcessor(pipeline: self, processorModel: cacheProcessorModel.model!)

        let promptTokenCount = tokens.count
        return AsyncThrowingStream<Prediction, Error> { continuation in
            var tokens = tokens
            let maxTokens = tokens.count + maxNewTokens
            while tokens.count < maxTokens {
                let timer = CodeTimer()
                let tokenSignpostState = self.signposter.beginInterval("Predict", id: self.signposter.makeSignpostID(), "Token #\(tokens.count)")
                let isPrompt = tokens.count == promptTokenCount

                // Do a forward pass of the model.
                var logits: MLMultiArray!
                for (i, chunk) in self.chunks.enumerated() {
                    // TODO: don't hardcode
                    let model = chunk.model(forTokenCount: tokens.count, newTokenCount: isPrompt ? 512 : 1)

                    // Wait for the KV cache to be asynchronously updated.
                    try await kvCacheProcessor.wait(forChunk: i)

                    let inputs = try await arrayStore.featureProvider(forChunk: i, model: model, tokens: tokens)
                    let options = MLPredictionOptions()
                    options.outputBackings = arrayStore.outputBackings(forChunk: i, model: model)

                    let predictState = self.signposter.beginInterval("Predict", id: self.signposter.makeSignpostID(), "Chunk \(i)")
                    let outputs = try await model.prediction(from: inputs, options: options)
                    self.signposter.endInterval("Predict", predictState)
                    arrayStore.update(outputs: outputs, forChunk: i) // Using output backings should make this ~a noop.

                    // Start asynchronously updating the KV cache for the next token prediction.
                    if #unavailable(macOS 15) {
                        kvCacheProcessor.submit(inputs: inputs, outputs: outputs, forChunk: i)
                    }

                    if outputs.featureNames.contains("logits") {
                        logits = outputs.featureValue(for: "logits")!.multiArrayValue!
                    }
                }

                let newToken =  try await self.logitProcessor.argmax(logits: logits)
                tokens.append(newToken)

                // Switch to the generation model before yielding so that
                // this is counted in the prompt processing time.
                if isPrompt {
                    // TODO: Support switching to larger context sizes.
                    let inputLength = self.chunks[0].generationModel!.sequenceLength()
                    let config = PipelineInferenceConfiguration(vocabSize: inferenceConfiguration.vocabSize,
                                                                inputLength: inputLength, contextLength: 512)
                    let resizeState = self.signposter.beginInterval("Resize Arrays")
                    await arrayStore.resize(for: self, with: config)
                    self.signposter.endInterval("Resize Arrays", resizeState)
                }

                continuation.yield(Prediction(newToken: newToken, allTokens: tokens, latency: timer.elapsed()))

                self.signposter.endInterval("Predict", tokenSignpostState, "\(newToken)")
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
            throw PipelineError.notFound
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
    var promptModel: MLModel?
    var generationModel: MLModel?

    convenience init?(url: URL, configuration: MLModelConfiguration) {
        guard let fileInfo = ChunkFileInfo(url: url) else { return nil }
        self.init(fileInfo: fileInfo, configuration: configuration)
    }

    init(fileInfo: ChunkFileInfo, configuration: MLModelConfiguration) {
        self.fileInfo = fileInfo
        self.configuration = configuration
    }

    func load() async {
        guard promptModel == nil else { return }

        if #available(macOS 15.0, *) {
            // TODO: Would be nice to load both in parallel if it helps on Sequoia after beta 1.
            let asset = try! MLModelAsset(url: fileInfo.url)
            promptModel = try! await MLModel.load(asset: asset,
                                                  configuration: configuration.withFunctionName("input_512_context_512"))
            generationModel = try! await MLModel.load(asset: asset,
                                                      configuration: configuration.withFunctionName("input_1_context_512").withInfrequentReshapes())
        } else {
            promptModel = try! MLModel(contentsOf: fileInfo.url, configuration: configuration)
            generationModel = promptModel
        }
    }

    func unload() {
        promptModel = nil
        generationModel = nil
    }

    func model(forTokenCount: Int, newTokenCount: Int) -> MLModel {
        if newTokenCount > generationModel!.sequenceLength() {
            return promptModel!
        }
        return generationModel!
    }
}

extension MLModel {
    func sequenceLength() -> Int {
        var name = ""
        for n in modelDescription.inputDescriptionsByName.keys {
            if n == "x" || n == "input_ids" {
                name = n
            }
        }
        return modelDescription.inputDescriptionsByName[name]!.multiArrayConstraint!.shape.last!.intValue
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
        guard
            var split = self.fileName.split(separator: ".").first?.split(separator: "_"),
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
    case unimplementedLongPrompt // Could support this, just not yet.
    case unsupportedInferenceConfiguration
    case cacheProcessorNotFound
    case notFound
    case ambiguousModelPath(possiblePrefixes: [String])
    case notImplementedError
}

struct Prediction {
    let newToken: Int
    let allTokens: [Int]
    let latency: Measurement<UnitDuration>
}

struct CodeTimer {
    let start = CFAbsoluteTimeGetCurrent()

    func elapsed() -> Measurement<UnitDuration> {
        let seconds = CFAbsoluteTimeGetCurrent() - start
        return Measurement(value: seconds, unit: .seconds)
    }
}

protocol Loadable {
    func load() async
    func unload()
}

extension LogitProcessor: Loadable {}
extension DeferredModel: Loadable {}
extension PipelineChunk: Loadable {}
