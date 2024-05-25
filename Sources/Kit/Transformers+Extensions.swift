import Foundation
import Tokenizers
import Hub

extension AutoTokenizer {
    /// Load the tokenizer from a cache when offline.
    /// Workaround until https://github.com/huggingface/swift-transformers/issues/39
    static func from(
        cached model: String,
        hubApi: HubApi = .shared
    ) async throws -> Tokenizer {
        // Network first since there is no cache invalidation.
        do {
            let config = LanguageModelConfigurationFromHub(modelName: model, hubApi: hubApi)
            guard let tokenizerConfig = try await config.tokenizerConfig else { throw TokenizerError.missingConfig }
            let tokenizerData = try await config.tokenizerData

            return try AutoTokenizer.from(tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
        } catch {
            let modelFolder = hubApi.localRepoLocation(.init(id: model))
            return try await from(modelFolder: modelFolder, hubApi: hubApi)
        }
    }
}

// Not public, so copied over.
enum TokenizerError : Error {
    case missingConfig
}

extension HubApi {
    static var defaultTokenLocation: URL {
        FileManager.default.homeDirectoryForCurrentUser.appending(path: ".cache/huggingface/token")
    }

    static func defaultToken() -> String? {
        if let envToken = ProcessInfo.processInfo.environment["HF_TOKEN"] {
            return envToken
        }
        return try? String(contentsOf: defaultTokenLocation, encoding: .utf8)
    }
}
