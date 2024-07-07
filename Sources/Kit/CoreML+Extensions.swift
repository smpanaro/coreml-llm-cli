import Foundation
import CoreML

extension MLComputeUnits {
     var debugName: String {
        switch self {
        case .all: return "All"
        case .cpuAndGPU: return "CPU+GPU"
        case .cpuAndNeuralEngine: return "CPU+ANE"
        case .cpuOnly: return "CPU"
        @unknown default: return "Unknown"
        }
    }
}

extension MLShapedArray where Scalar == Float16 {
    @available(macOS 15, *)
    static func emptyIOSurfaceArray(shape: [Int]) -> MLShapedArray<Float16> {
        let pixelBuffer = CVPixelBuffer.emptyIOSurfaceBuffer(shape: shape)
        return MLShapedArray<Float16>(mutating: pixelBuffer!, shape: shape)
    }
}

extension MLMultiArray {
    class func emptyIOSurfaceArray(shape: [Int]) -> MLMultiArray? {
        guard let pixelBuffer = CVPixelBuffer.emptyIOSurfaceBuffer(shape: shape) else {
            return nil
        }
        return MLMultiArray(pixelBuffer: pixelBuffer, shape: shape.map({ $0 as NSNumber }))
    }

    @available(macOS 15, *)
    func toTensor() -> MLTensor {
        MLTensor(MLShapedArray<Float16>(self))
    }
}

extension CVPixelBuffer {
    class func emptyIOSurfaceBuffer(shape: [Int]) -> CVPixelBuffer? {
        guard
            shape.count > 0,
            let width = shape.last
        else { return nil }
        let height = shape[0..<shape.count-1].reduce(1, { $0 * $1 })

        let attributes = [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary
        var pixelBuffer: CVPixelBuffer?
        guard kCVReturnSuccess == CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_OneComponent16Half,
            attributes,
            &pixelBuffer)
        else { return nil }
        guard let pixelBuffer = pixelBuffer else { return nil }

        precondition(kCVReturnSuccess == CVPixelBufferLockBaseAddress(pixelBuffer, .init(rawValue: 0)),
                     "Failed to lock pixel buffer base address")
        memset(CVPixelBufferGetBaseAddress(pixelBuffer), 0, CVPixelBufferGetDataSize(pixelBuffer))
        precondition(kCVReturnSuccess == CVPixelBufferUnlockBaseAddress(pixelBuffer, .init(rawValue: 0)),
                     "Failed to unlock pixel buffer base address")

        return pixelBuffer
    }
}

extension MLPredictionOptions {
    /// Returns the keys of the float16 PixelBuffer-backed MLMultiArray
    /// backings that were ignored by CoreML.
    func ignoredOutputBackingKeys(_ outputs: MLFeatureProvider) -> [String] {
        var ignored = [String]()
        outputBackings.forEach { name, backing in
            guard let multiArrayValue = outputs.featureValue(for: name)?.multiArrayValue,
                  multiArrayValue.dataType == .float16
            else { preconditionFailure("no float16 multiArray output for backing name: \(name)") }

            guard let outputPixelBuffer = multiArrayValue.pixelBuffer
            else { preconditionFailure("output array is not pixel buffer backed: \(name)") }

            guard let backingPixelBuffer = (backing as? MLMultiArray)?.pixelBuffer
            else { preconditionFailure("output backing is not pixel buffer backed: \(name)")}

            if outputPixelBuffer !== backingPixelBuffer {
                ignored.append(name)
            }
        }

        return ignored
    }
}

extension MLModelConfiguration {
    func withInfrequentReshapes() -> Self {
        let new = copy() as! Self
        if #available(macOS 14.4, *) {
            new.optimizationHints = MLOptimizationHints()
            new.optimizationHints.reshapeFrequency = .infrequent
        }
        return new
    }

    func withFunctionName(_ name: String?) -> Self {
        let new = copy() as! Self
        if #available(macOS 15, *) {
            new.functionName = name
        }
        return new
    }
}
