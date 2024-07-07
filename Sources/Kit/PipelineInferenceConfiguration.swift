import Foundation
import CoreML

struct PipelineInferenceConfiguration {
    let vocabSize: Int
    let inputLength: Int // query length
    let contextLength: Int // key + value length

    var cacheLength: Int {
        contextLength - inputLength
    }
}

extension PipelineInferenceConfiguration {
    init?(from models: [MLModel]) {
        guard models.count > 2 else { return nil }

        guard let first = models.first, let last = models.last
        else { return nil }
        let innerModels = models[1..<models.count-1]

        let inputIDs = first.modelDescription.inputDescriptionsByName["input_ids"]
        self.inputLength = inputIDs!.multiArrayConstraint!.shape.last!.intValue

        let logits = last.modelDescription.outputDescriptionsByName["logits"]
        self.vocabSize = logits!.multiArrayConstraint!.shape.last!.intValue

        guard let firstInnerModelInputs = innerModels.first?.modelDescription.inputDescriptionsByName
        else { return nil }

        // Use k cache for Sonoma models and Sequoia generation models. x for Sequoia prompt models.
        let contextTensorDescription = firstInnerModelInputs["k_cache_0"] ?? firstInnerModelInputs["x"]
        self.contextLength = self.inputLength + contextTensorDescription!.multiArrayConstraint!.shape.last!.intValue
    }
}

