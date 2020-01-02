// Copyright 2020, Emmanouil Antonios Platanios. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

import TensorFlow

public enum Modality {
  case image, number
}

// Transform colors (e.g., add/remove color to/from image).
// Rotate image.

// Perceptive
// ----------
// Rotation: Image => Image
// ColorMap: Image => Image
//
// Reasoning
// ---------
// Identity
// Addition
//
// Inverse??? (maybe should be treated as a functor?)

public enum Problem {
  case identity
  case rotation(Float)
  case colorMap(Tensor<Float>)
}

//===------------------------------------------------------------------------------------------===//
// Problem Compilers
//===------------------------------------------------------------------------------------------===//

public protocol ProblemCompiler: Differentiable {
  var problemEmbeddingSize: Int { get }

  @differentiable
  func perceptionContext(forProblem problem: Problem) -> Tensor<Float>

  @differentiable
  func reasoningContext(forProblem problem: Problem) -> Tensor<Float>

  @differentiable
  func generationContext(forProblem problem: Problem) -> Tensor<Float>
}

public struct LinearProblemCompiler: ProblemCompiler {
  @noDerivative public let problemEmbeddingSize: Int

  public var identityEmbedding: Tensor<Float>
  public var rotationEmbedding: Tensor<Float>
  public var colorMapEmbedding: Tensor<Float>

  public init(problemEmbeddingSize: Int, initializerStandardDeviation: Float = 0.02) {
    self.problemEmbeddingSize = problemEmbeddingSize
    let initializer = truncatedNormalInitializer(
      standardDeviation: Tensor<Float>(initializerStandardDeviation))
    self.identityEmbedding = initializer([1, problemEmbeddingSize])
    self.rotationEmbedding = initializer([2, problemEmbeddingSize])
    self.colorMapEmbedding = initializer([3, problemEmbeddingSize])
  }

  @differentiable
  public func perceptionContext(forProblem problem: Problem) -> Tensor<Float> {
    return identityEmbedding
  }

  @differentiable
  public func reasoningContext(forProblem problem: Problem) -> Tensor<Float> {
    return identityEmbedding
  }

  @differentiable
  public func generationContext(forProblem problem: Problem) -> Tensor<Float> {
    switch problem {
    case .identity:
      return identityEmbedding
    case let .rotation(degrees):
      return (rotationEmbedding[0] * degrees / 360 + rotationEmbedding[1]).rankLifted()
    case let .colorMap(factors):
      return matmul(factors.rankLifted(), colorMapEmbedding)
    }
  }
}

//===------------------------------------------------------------------------------------------===//
// Tasks
//===------------------------------------------------------------------------------------------===//

public protocol Task {
  // mutating func update<A: Architecture, O: Optimizer>(
  //   architecture: inout A,
  //   using optimizer: inout O
  // ) -> Float where O.Model == A
  mutating func update<O: Optimizer>(
    architecture: inout ConvolutionalArchitecture,
    using optimizer: inout O
  ) -> Float where O.Model == ConvolutionalArchitecture
}

public struct Example: KeyPathIterable {
  public var input: Tensor<Float>
  public var output: Tensor<Float>

  public let degrees: Float

  public init(input: Tensor<Float>, output: Tensor<Float>, degrees: Float = 0.0) {
    self.input = input
    self.output = output
    self.degrees = degrees
  }
}

public struct IdentityTask: Task {
  public let srcModality: Modality
  public let tgtModality: Modality
  public let dataset: Dataset
  public let randomRotations: Bool

  public let problem: Problem = .identity

  internal typealias ExampleIterator = IndexingIterator<Array<Int>>
  internal typealias RepeatExampleIterator = ShuffleIterator<RepeatIterator<ExampleIterator>>
  internal typealias TrainDataIterator = PrefetchIterator<BatchIterator<MapIterator<RepeatExampleIterator, Example>>>

  internal var trnExamplesIterator: TrainDataIterator

  public init(
    srcModality: Modality,
    tgtModality: Modality,
    dataset: Dataset,
    randomRotations: Bool = false,
    randomSeed: Int64 = 123456789
  ) {
    var generator = PhiloxRandomNumberGenerator(seed: randomSeed)
    self.srcModality = srcModality
    self.tgtModality = tgtModality
    self.dataset = dataset
    self.randomRotations = randomRotations
    self.trnExamplesIterator = dataset.partitions[.train]!
      .makeIterator()
      .repeated()
      .shuffled(bufferSize: 1000)
      .map { index -> Example in
        let srcNumber = dataset.numbers[index]
        let tgtNumber = srcNumber
        let input = { () -> Tensor<Float> in
          switch (srcModality) {
          case .image:
            if randomRotations {
              let degrees = Float.random(in: 0..<360, using: &generator)
              return rotate(image: dataset.images[index], degrees: degrees)
            } else {
              return dataset.images[index]
            }
          case .number:
            return Tensor<Float>(dataset.numbers[index])
          }
        }()
        let output = { () -> Tensor<Float> in
          switch (tgtModality) {
          case .image:
            let exampleIndices = dataset.numberImageIndices[.train]![tgtNumber]!
            let index = exampleIndices.randomElement(using: &generator)!
            return Tensor<Float>(dataset.images[index])
          case .number:
            return Tensor<Float>(tgtNumber)
          }
        }()
        return Example(input: input, output: output)
      }
      .batched(batchSize: batchSize)
      .prefetched(count: 2)
  }

  // public mutating func update<A: Architecture, O: Optimizer>(
  //   architecture: inout A,
  //   using optimizer: inout O
  // ) -> Float where O.Model == A {
  public mutating func update<O: Optimizer>(
    architecture: inout ConvolutionalArchitecture,
    using optimizer: inout O
  ) -> Float where O.Model == ConvolutionalArchitecture {
    let batch = trnExamplesIterator.next()!
    let problem = self.problem
    return withLearningPhase(.training) {
      let (loss, gradient) = { () -> (Tensor<Float>, ConvolutionalArchitecture.TangentVector) in
        switch (srcModality, tgtModality) {
        case (.image, .image):
          return valueWithGradient(at: architecture) {
            l2Loss(
              predicted: $0.generateImage(forImage: batch.input, problem: problem),
              expected: batch.output,
              reduction: { $0.mean() })
          }
        case (.image, .number):
          return valueWithGradient(at: architecture) {
            softmaxCrossEntropy(
              logits: $0.generateNumber(forImage: batch.input, problem: problem),
              labels: Tensor<Int32>(batch.output),
              reduction: { $0.mean() })
          }
        case (.number, .image):
          return valueWithGradient(at: architecture) {
            l2Loss(
              predicted: $0.generateImage(forNumber: batch.input, problem: problem),
              expected: batch.output,
              reduction: { $0.mean() })
          }
        case (.number, .number):
          return valueWithGradient(at: architecture) {
            softmaxCrossEntropy(
              logits: $0.generateNumber(forNumber: batch.input, problem: problem),
              labels: Tensor<Int32>(batch.output),
              reduction: { $0.mean() })
          }
        }
      }()
      optimizer.update(&architecture, along: gradient)
      return loss.scalarized()
    }
  }
}

public struct RotationTask: Task {
  public let dataset: Dataset

  private typealias ExampleIterator = IndexingIterator<Array<Int>>
  private typealias RepeatExampleIterator = ShuffleIterator<RepeatIterator<ExampleIterator>>
  private typealias TrainDataIterator = PrefetchIterator<BatchIterator<MapIterator<RepeatExampleIterator, Example>>>

  private var trnExamplesIterator: TrainDataIterator

  public init(dataset: Dataset, randomSeed: Int64 = 123456789) {
    var generator = PhiloxRandomNumberGenerator(seed: randomSeed)
    self.dataset = dataset
    self.trnExamplesIterator = dataset.partitions[.train]!
      .makeIterator()
      .repeated()
      .shuffled(bufferSize: 1000)
      .map { index -> Example in
        let degrees = Float.random(in: 0..<360, using: &generator)
        let input = dataset.images[index]
        let output = rotate(image: input, degrees: degrees)
        return Example(input: input, output: output, degrees: degrees)
      }
      .batched(batchSize: batchSize)
      .prefetched(count: 2)
  }

  // public mutating func update<A: Architecture, O: Optimizer>(
  //   architecture: inout A,
  //   using optimizer: inout O
  // ) -> Float where O.Model == A {
  public mutating func update<O: Optimizer>(
    architecture: inout ConvolutionalArchitecture,
    using optimizer: inout O
  ) -> Float where O.Model == ConvolutionalArchitecture {
    let batch = trnExamplesIterator.next()!
    let problem = Problem.rotation(batch.degrees)
    return withLearningPhase(.training) {
      let (loss, gradient) = valueWithGradient(at: architecture) {
        l2Loss(
          predicted: $0.generateImage(forImage: batch.input, problem: problem),
          expected: batch.output,
          reduction: { $0.mean() })
      }
      optimizer.update(&architecture, along: gradient)
      return loss.scalarized()
    }
  }
}
