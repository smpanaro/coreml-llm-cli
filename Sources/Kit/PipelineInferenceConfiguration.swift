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

        let logits = last.modelDescription.outputDescriptionsByName["logits"] ?? last.modelDescription.outputDescriptionsByName["logits_0"]
        self.vocabSize = logits!.multiArrayConstraint!.shape.last!.intValue

        guard let firstInnerModel = innerModels.first
        else { return nil }

        let firstKeyCache = firstInnerModel.modelDescription.inputDescriptionsByName["k_cache_0"]
        self.contextLength = self.inputLength + firstKeyCache!.multiArrayConstraint!.shape.last!.intValue
    }
}

