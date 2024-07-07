import Foundation
import Tokenizers

/// TextGenerator uses an LLM `ModelPipeline` to generate text.
class TextGenerator {
    let pipeline: ModelPipeline
    let tokenizer: Tokenizer
    // var strategy: Strategy = .argmax // etc.

    init(pipeline: ModelPipeline, tokenizer: Tokenizer) {
        self.pipeline = pipeline
        self.tokenizer = tokenizer
    }

    func generate(text: String, maxNewTokens: Int) async throws {

        let loadTimer = CodeTimer()
        try await pipeline.load()
        let loadDuration = loadTimer.elapsed()

        let tokens = tokenizer.encode(text: text)

        var predictions = [Prediction]()
        tokens.forEach { print($0, terminator: " ") }
        print("|", terminator: " ")
        fflush(stdout)

        for try await prediction in try pipeline.predict(tokens: tokens, maxNewTokens: maxNewTokens) {
            predictions.append(prediction)
            print(prediction.newToken, terminator: " ")
            fflush(stdout)
        }
        print("\n")

        print(tokenizer.decode(tokens: predictions.last?.allTokens ?? tokens))
        print()

        print("Compile + Load: \(loadDuration.converted(to: .seconds).value.formatted(.number.precision(.fractionLength(2)))) sec")

        let numberFormat = FloatingPointFormatStyle<Double>.number.precision(.fractionLength(2))

        if #available(macOS 15, *) {
            let promptPrediction = predictions.removeFirst()
            print("Prompt        :", terminator: " ")
            let promptDuration = promptPrediction.latency.converted(to: .milliseconds)
//            let promptAverage = promptDuration / 512.0 // Prompt always process 512 tokens.
            let promptThroughput = (512.0 / promptDuration.converted(to: .seconds))
            print("\(promptDuration.value.formatted(numberFormat)) sec")
//            print("                \(promptAverage.value.formatted(numberFormat)) ms / token")
            print("                \(promptThroughput.value.formatted(numberFormat)) token / sec")
        }

        print("Generate      :", terminator: " ")
        let latencies = predictions.map { $0.latency.converted(to: .milliseconds).value }
        let average = Measurement(value: latencies.mean(), unit: UnitDuration.milliseconds)
        let stdev = Measurement(value: latencies.stdev(), unit: UnitDuration.milliseconds)
        print("\(average.value.formatted(numberFormat)) +/- \(stdev.value.formatted(numberFormat)) ms / token")

        let throughputs = predictions.map { 1 / $0.latency.converted(to: .seconds).value }
        let averageThroughput = Measurement(value: throughputs.mean(), unit: UnitDuration.seconds)
        let stdevThroughput = Measurement(value: throughputs.stdev(), unit: UnitDuration.seconds)
        print("                \(averageThroughput.value.formatted(numberFormat)) +/- \(stdevThroughput.value.formatted(numberFormat)) token / sec")
    }
}

