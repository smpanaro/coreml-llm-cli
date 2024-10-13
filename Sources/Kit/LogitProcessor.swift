import Foundation
import CoreML
import OSLog

/// Pick the next token from the provided logits.
class LogitProcessor {
    let model: DeferredModel

    let signposter = OSSignposter(subsystem: "com.stephenpanaro.llm-cli", category: "LogitProcessor")

    init(model: DeferredModel) {
        self.model = model
    }

    func load() {
        self.model.load()
    }

    func unload() {
        self.model.unload()
    }

//    func multinomial(logits: MLMultiArray) async throws -> Int {
//    }

    func argmax(logits: [MLMultiArray], index: Int? = nil) async throws -> Int {
        let state = signposter.beginInterval("Sample", "Argmax")
        defer { signposter.endInterval("Sample", state) }

        let outputs = try await predict(logits: logits)
        let argmaxArray = MLShapedArray<Int32>(outputs.featureValue(for: "argmax")!.multiArrayValue!)
        let prediction = argmaxArray[0, index ?? (argmaxArray.shape[1] - 1)].scalar ?? -1
        return Int(prediction)
    }

    fileprivate func predict(logits: [MLMultiArray]) async throws -> MLFeatureProvider {
        let keysValues = logits.enumerated().map { (index, array) in
            return (logits.count == 1 ? "logits" : "logits_\(index)", array)
        }
        let inputs = try MLDictionaryFeatureProvider(
            dictionary: Dictionary(keysValues, uniquingKeysWith: { (first, _) in first })
        )
        return try await model.model!.prediction(from: inputs)
    }
}
