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

public protocol Architecture: Differentiable, KeyPathIterable
where TangentVector: KeyPathIterable {
  associatedtype ProblemCompiler: MNIST.ProblemCompiler

  @differentiable var problemCompiler: ProblemCompiler { get set }

  /// - Returns: Tensor with shape `[batchSize, hiddenSize]`.
  @differentiable
  func perceive(image: Tensor<Float>) -> Tensor<Float>

  /// - Returns: Tensor with shape `[batchSize, hiddenSize]`.
  @differentiable
  func perceive(number: Tensor<Float>) -> Tensor<Float>

  @differentiable
  func reason(over input: Tensor<Float>, context: Tensor<Float>) -> Tensor<Float>

  @differentiable
  func generateImage(reasoningOutput: Tensor<Float>) -> Tensor<Float>

  @differentiable
  func generateNumber(reasoningOutput: Tensor<Float>) -> Tensor<Float>
}

extension Architecture {
  // @differentiable
  public func generateImage(forImage image: Tensor<Float>, problem: Problem) -> Tensor<Float> {
    let context = problemCompiler.compile(problem: problem)
    let latent = reason(over: perceive(image: image), context: context)
    return generateImage(reasoningOutput: latent)
  }

  // @differentiable
  public func generateImage(forNumber number: Tensor<Float>, problem: Problem) -> Tensor<Float> {
    let context = problemCompiler.compile(problem: problem)
    let latent = reason(over: perceive(number: number), context: context)
    return generateImage(reasoningOutput: latent)
  }

  // @differentiable
  public func generateNumber(forImage image: Tensor<Float>, problem: Problem) -> Tensor<Float> {
    let context = problemCompiler.compile(problem: problem)
    let latent = reason(over: perceive(image: image), context: context)
    return generateNumber(reasoningOutput: latent)
  }

  // @differentiable
  public func generateNumber(forNumber number: Tensor<Float>, problem: Problem) -> Tensor<Float> {
    let context = problemCompiler.compile(problem: problem)
    let latent = reason(over: perceive(number: number), context: context)
    return generateNumber(reasoningOutput: latent)
  }
}

public struct ConvolutionalArchitecture: Architecture {
  @noDerivative public let hiddenSize: Int

  public var problemCompiler: LinearProblemCompiler
  public var numberEmbeddings: Tensor<Float>
  public var percConv1: Conv2D<Float>
  public var percPool1: MaxPool2D<Float>
  public var percConv2: Conv2D<Float>
  public var percPool2: MaxPool2D<Float>
  public var percFlatten: Flatten<Float>
  public var percFC1: Dense<Float>
  public var percFC2: Dense<Float>
  public var genFC1: Dense<Float>
  public var genFC2: Dense<Float>
  public var genReshape: Reshape<Float>
  public var genTransposedConv1: TransposedConv2D<Float>
  public var genTransposedConv2: TransposedConv2D<Float>
  public var reasoningLayer: ContextualizedLayer<Dense<Float>, Dense<Float>>

  public init(
    hiddenSize: Int,
    problemCompiler: LinearProblemCompiler,
    initializerStandardDeviation: Float = 0.02
  ) {
    self.hiddenSize = hiddenSize
    self.problemCompiler = problemCompiler
    self.numberEmbeddings = truncatedNormalInitializer(
      standardDeviation: Tensor<Float>(initializerStandardDeviation)
    )([10, hiddenSize])
    self.percConv1 = Conv2D<Float>(filterShape: (5, 5, 3, 32), padding: .same, activation: gelu)
    self.percPool1 = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    self.percConv2 = Conv2D<Float>(filterShape: (5, 5, 32, 64), padding: .same, activation: gelu)
    self.percPool2 = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    self.percFlatten = Flatten<Float>()
    self.percFC1 = Dense<Float>(inputSize: 7 * 7 * 64, outputSize: 1024, activation: gelu)
    self.percFC2 = Dense<Float>(inputSize: 1024, outputSize: hiddenSize)
    self.genFC1 = Dense<Float>(inputSize: hiddenSize, outputSize: 1024, activation: gelu)
    self.genFC2 = Dense<Float>(inputSize: 1024, outputSize: 7 * 7 * 64, activation: gelu)
    self.genReshape = Reshape<Float>(shape: [-1, 7, 7, 64])
    self.genTransposedConv1 = TransposedConv2D<Float>(
      filterShape: (5, 5, 32, 64),
      strides: (2, 2),
      padding: .same,
      activation: gelu)
    self.genTransposedConv2 = TransposedConv2D<Float>(
      filterShape: (5, 5, 3, 32),
      strides: (2, 2),
      padding: .same)
    let reasoningLayerBase = Dense<Float>(
      inputSize: hiddenSize,
      outputSize: hiddenSize,
      weightInitializer: truncatedNormalInitializer(
        standardDeviation: Tensor(initializerStandardDeviation)))
    self.reasoningLayer = ContextualizedLayer(
      base: reasoningLayerBase,
      generator: Dense<Float>(
        inputSize: problemCompiler.problemEmbeddingSize,
        outputSize: reasoningLayerBase.parameterCount,
        weightInitializer: truncatedNormalInitializer(
          standardDeviation: Tensor(initializerStandardDeviation))))
  }

  /// - Returns: Tensor with shape `[batchSize, hiddenSize]`.
  @differentiable
  public func perceive(image: Tensor<Float>) -> Tensor<Float> {
    let convolved = image.sequenced(through: percConv1, percPool1, percConv2, percPool2)
    return convolved.sequenced(through: percFlatten, percFC1, percFC2)
  }

  /// - Returns: Tensor with shape `[batchSize, hiddenSize]`.
  @differentiable
  public func perceive(number: Tensor<Float>) -> Tensor<Float> {
    let indices = withoutDerivative(at: number, in: Tensor<Int32>.init)
    return numberEmbeddings.gathering(atIndices: indices)
  }

  @differentiable
  public func reason(over input: Tensor<Float>, context: Tensor<Float>) -> Tensor<Float> {
    reasoningLayer(ContextualizedInput(input: input, context: context))
  }

  @differentiable
  public func generateImage(reasoningOutput: Tensor<Float>) -> Tensor<Float> {
    let transformed = reasoningOutput.sequenced(through: genFC1, genFC2, genReshape)
    let images = transformed.sequenced(through: genTransposedConv1, genTransposedConv2)
    return softmax(images)
  }

  @differentiable
  public func generateNumber(reasoningOutput: Tensor<Float>) -> Tensor<Float> {
    matmul(reasoningOutput, transposed: false, numberEmbeddings, transposed: true)
  }
}

extension ConvolutionalArchitecture {
  @differentiable
  public func generateImage(forImage image: Tensor<Float>, problem: Problem) -> Tensor<Float> {
    let context = problemCompiler.compile(problem: problem)
    let latent = reason(over: perceive(image: image), context: context)
    return generateImage(reasoningOutput: latent)
  }

  @differentiable
  public func generateImage(forNumber number: Tensor<Float>, problem: Problem) -> Tensor<Float> {
    let context = problemCompiler.compile(problem: problem)
    let latent = reason(over: perceive(number: number), context: context)
    return generateImage(reasoningOutput: latent)
  }

  @differentiable
  public func generateNumber(forImage image: Tensor<Float>, problem: Problem) -> Tensor<Float> {
    let context = problemCompiler.compile(problem: problem)
    let latent = reason(over: perceive(image: image), context: context)
    return generateNumber(reasoningOutput: latent)
  }

  @differentiable
  public func generateNumber(forNumber number: Tensor<Float>, problem: Problem) -> Tensor<Float> {
    let context = problemCompiler.compile(problem: problem)
    let latent = reason(over: perceive(number: number), context: context)
    return generateNumber(reasoningOutput: latent)
  }
}
