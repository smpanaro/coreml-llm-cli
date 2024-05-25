import Foundation

extension Array where Element: FloatingPoint {
    func sum() -> Element {
        return self.reduce(0, +)
    }

    func mean() -> Element {
        return self.sum() / Element(self.count)
    }

    func stdev() -> Element {
        let mean = self.mean()
        let v = self.reduce(0, { $0 + ($1-mean)*($1-mean) })
        return sqrt(v / (Element(self.count) - 1))
    }
}

extension AsyncThrowingStream {
    /// Initialize a stream that runs its body closure asynchronously.
    public init(
        _ elementType: Element.Type = Element.self,
        bufferingPolicy limit: AsyncThrowingStream<Element, Failure>.Continuation.BufferingPolicy = .unbounded,
        _ build: @Sendable @escaping (AsyncThrowingStream<Element, Failure>.Continuation) async throws -> Void
    ) where Failure == Error {
        self = AsyncThrowingStream(elementType, bufferingPolicy: limit) { continuation in
            let task = Task {
                try await build(continuation)
            }

            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }
}
