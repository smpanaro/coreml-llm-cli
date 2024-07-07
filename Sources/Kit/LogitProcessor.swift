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

    func argmax(logits: MLMultiArray) async throws -> Int {
        let state = signposter.beginInterval("Sample", "Argmax")
        defer { signposter.endInterval("Sample", state) }

        let outputs = try await predict(logits: logits)
        let argmaxArray = MLShapedArray<Int32>(outputs.featureValue(for: "argmax")!.multiArrayValue!)
        let prediction = argmaxArray[0, argmaxArray.shape[1] - 1].scalar ?? -1
        return Int(prediction)
    }

    fileprivate func predict(logits: MLMultiArray) async throws -> MLFeatureProvider {
        let inputs = try MLDictionaryFeatureProvider(dictionary: [
            "logits": await resizeLogits(logits: logits),
        ])
        return try await model.model!.prediction(from: inputs)
    }

    fileprivate func resizeLogits(logits: MLMultiArray) async -> MLMultiArray {
        // macOS 14 is always length 64.
        guard #available(macOS 15, *) else { return logits }
        // TODO: Don't hardcode this.
        guard ![64,511,1,512].contains(logits.shape[1]) else { return logits }

        let tensor = MLTensor(MLShapedArray<Float16>(logits))
        let (batch, sequence, vocab) = (tensor.shape[0], tensor.shape[1], tensor.shape[2])
        let last64 = sequence >= 64 ?
            tensor[0..<batch, (sequence-64)..., ...] :
            MLTensor(zeros: [batch, 64-sequence, vocab], scalarType: Float16.self).concatenated(with: tensor, alongAxis: 1)
        let newLogits = await last64.shapedArray(of: Float16.self)

        return MLMultiArray(newLogits)
    }
}
