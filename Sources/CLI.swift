import Foundation
import ArgumentParser
import Tokenizers
import Hub
import CoreML

@main
struct CLI: AsyncParsableCommand {
    @Option(help: "Huggingface repo ID. e.g. smpanaro/Llama-2-7b-CoreML")
    var repoID: String? = nil

    @Option(
        help: "The directory containing the model's mlmodelc files.",
        completion: .file(), transform: URL.init(fileURLWithPath:))
    var localModelDirectory: URL?

    @Option(help: "The model filename prefix, to differentiate when there are multiple models in a folder.")
    var localModelPrefix: String?

    @Option(help: "KV cache processor model filename, located in the model directory.")
    var cacheProcessorModelName: String = "cache-processor.mlmodelc"

    @Option(help: "Logit processor model filename, located in the model directory.")
    var logitProcessorModelName: String = "logit-processor.mlmodelc"

    @Option(help: "Tokenizer name on huggingface.")
    var tokenizerName: String = "pcuenq/Llama-2-7b-chat-coreml"

    @Argument(help: "Input text.")
    var inputText: String = "Several species of shrub of the genus Coffea produce the berries from which coffee is extracted. The two main species commercially cultivated are Coffea canephora"

    @Option(help: "Maximum number of new tokens to generate.")
    var maxNewTokens: Int = 60

    mutating func run() async throws {
        var modelDirectory = localModelDirectory
        if let repoID {
            modelDirectory = try await downloadModel(repoID: repoID)
        }

        guard let modelDirectory else {
            print("Either --repoID or --localModelDirectory must be provided.")
            return
        }

        let pipeline = try ModelPipeline.from(
            folder: modelDirectory,
            modelPrefix: localModelPrefix,
            cacheProcessorModelName: cacheProcessorModelName,
            logitProcessorModelName: logitProcessorModelName
            // For debugging.
//            primaryCompute: .cpuOnly
//            chunkLimit: 1
        )
        print(pipeline)

        let tokenizer = try await AutoTokenizer.from(cached: tokenizerName, hubApi: .init(hfToken: HubApi.defaultToken()))

        let generator = TextGenerator(pipeline: pipeline, tokenizer: tokenizer)
        try await generator.generate(text: inputText, maxNewTokens: maxNewTokens)
    }

    /// Download a model and return the local directory URL.
    func downloadModel(repoID: String) async throws -> URL {
        let hub = HubApi(hfToken: HubApi.defaultToken())
        let repo = Hub.Repo(id: repoID, type: .models)

        let mlmodelcs = ["Llama*.mlmodelc/*", "logit*", "cache*"]
        let filenames = try await hub.getFilenames(from: repo, matching: mlmodelcs)

        let localURL = hub.localRepoLocation(repo)
        let localFileURLs = filenames.map {
            localURL.appending(component: $0)
        }
        let anyNotExists = localFileURLs.filter {
            !FileManager.default.fileExists(atPath: $0.path(percentEncoded: false))
        }.count > 0

        // swift-transformers doesn't offer a way to know if we're up to date.
        let newestTimestamp = localFileURLs.filter {
            FileManager.default.fileExists(atPath: $0.path(percentEncoded: false))
        }.compactMap {
           let attrs = try! FileManager.default.attributesOfItem(atPath: $0.path(percentEncoded: false))
           return attrs[.modificationDate] as? Date
        }.max() ?? Date.distantFuture
        let lastUploadDate = Date(timeIntervalSince1970: 1723688450)
        let isStale = repoID == "smpanaro/Llama-2-7b-coreml" && newestTimestamp < lastUploadDate

        // I would rather not delete things automatically.
        if isStale {
            print("⚠️ You have an old model downloaded. Please move the following directory to the Trash and try again:")
            print(localURL.path())
            throw CLIError.staleFiles
        }

        let needsDownload = anyNotExists || isStale
        guard needsDownload else { return localURL }

        print("Downloading from \(repoID)...")
        if filenames.count == 0 {
            throw CLIError.noModelFilesFound
        }

        let downloadDir = try await hub.snapshot(from: repo, matching: mlmodelcs) { progress in
            let percent = progress.fractionCompleted * 100
            if !progress.isFinished {
                print("\(percent.formatted(.number.precision(.fractionLength(0))))%", terminator: "\r")
                fflush(stdout)
            }
        }
        print("Done.")
        print("Downloaded to \(downloadDir.path())")
        return downloadDir
    }
}

enum CLIError: Error {
    case noModelFilesFound
    case staleFiles
}
