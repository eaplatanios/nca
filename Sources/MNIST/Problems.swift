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

public enum Problem {
  case identity
  case moduloAdd1
  indirect case inverse(Problem)
}

//===------------------------------------------------------------------------------------------===//
// Tasks
//===------------------------------------------------------------------------------------------===//

public struct Task {
  public let srcModality: Modality
  public let tgtModality: Modality
  public let problem: Problem
  public let dataset: Dataset

  private typealias ExampleIterator = IndexingIterator<Array<Int>>
  private typealias RepeatExampleIterator = ShuffleIterator<RepeatIterator<ExampleIterator>>
  private typealias TrainDataIterator = PrefetchIterator<BatchIterator<MapIterator<RepeatExampleIterator, Example>>>

  private var trnExamplesIterator: TrainDataIterator

  public init(
    srcModality: Modality,
    tgtModality: Modality,
    problem: Problem,
    dataset: Dataset,
    randomSeed: Int64 = 123456789
  ) {
    var generator = PhiloxRandomNumberGenerator(seed: randomSeed)
    self.srcModality = srcModality
    self.tgtModality = tgtModality
    self.problem = problem
    self.dataset = dataset
    self.trnExamplesIterator = dataset.partitions[.test]!
      .makeIterator()
      .repeated()
      .shuffled(bufferSize: 1000)
      .map { index -> Example in
        let srcNumber = dataset.numbers[index]
        let tgtNumber = target(for: srcNumber, problem: problem)
        let input = { () -> Tensor<Float> in
          switch (srcModality) {
          case .image: return dataset.images[index]
          case .number: return Tensor<Float>(dataset.numbers[index])
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

internal func target(for source: Float, problem: Problem) -> Float {
  switch (problem) {
    case .identity: return source
    case .moduloAdd1: return (source + 1).truncatingRemainder(dividingBy: 10)
    case .inverse(.identity): return source
    case .inverse(.moduloAdd1): return (source - 1).truncatingRemainder(dividingBy: 10)
    case let .inverse(.inverse(baseProblem)): return target(for: source, problem: baseProblem)
  }
}

//===------------------------------------------------------------------------------------------===//
// Problem Compilers
//===------------------------------------------------------------------------------------------===//

public protocol ProblemCompiler: Differentiable {
  var problemEmbeddingSize: Int { get }

  @differentiable
  func compile(problem: Problem) -> Tensor<Float>
}

public struct LinearProblemCompiler: ProblemCompiler {
  @noDerivative public let problemEmbeddingSize: Int

  public var identityEmbedding: Tensor<Float>
  public var inverseEmbedding: Tensor<Float>
  public var moduloAdd1Embedding: Tensor<Float>

  public init(problemEmbeddingSize: Int, initializerStandardDeviation: Float = 0.02) {
    self.problemEmbeddingSize = problemEmbeddingSize
    let initializer = truncatedNormalInitializer(
      standardDeviation: Tensor<Float>(initializerStandardDeviation))
    self.identityEmbedding = initializer([1, problemEmbeddingSize])
    self.inverseEmbedding = initializer([problemEmbeddingSize, problemEmbeddingSize])
    self.moduloAdd1Embedding = initializer([1, problemEmbeddingSize])
  }

  @differentiable
  public func compile(problem: Problem) -> Tensor<Float> {
    switch problem {
    case .identity: return identityEmbedding
    case .moduloAdd1: return moduloAdd1Embedding
    case let .inverse(baseProblem):
      let baseProblemEmbedding = compile(problem: baseProblem)
      return matmul(baseProblemEmbedding, inverseEmbedding)
    }
  }
}
