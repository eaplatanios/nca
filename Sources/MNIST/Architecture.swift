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

import Core
import TensorFlow

public protocol Architecture: Differentiable, KeyPathIterable
where TangentVector: KeyPathIterable {
  associatedtype ProblemCompiler: MNIST.ProblemCompiler

  @differentiable var problemCompiler: ProblemCompiler { get set }

  /// - Returns: Tensor with shape `[batchSize, hiddenSize]`.
  @differentiable
  func perceive(image: Tensor<Float>, context: Tensor<Float>) -> Tensor<Float>

  /// - Returns: Tensor with shape `[batchSize, hiddenSize]`.
  @differentiable
  func perceive(number: Tensor<Float>, context: Tensor<Float>) -> Tensor<Float>

  @differentiable
  func reason(over input: Tensor<Float>, context: Tensor<Float>) -> Tensor<Float>

  @differentiable
  func generateImage(reasoningOutput: Tensor<Float>, context: Tensor<Float>) -> Tensor<Float>

  @differentiable
  func generateNumber(reasoningOutput: Tensor<Float>, context: Tensor<Float>) -> Tensor<Float>
}

extension Architecture {
  // @differentiable
  public func generateImage(forImage image: Tensor<Float>, problem: Problem) -> Tensor<Float> {
    let perceptionContext = problemCompiler.perceptionContext(forProblem: problem)
    let reasoningContext = problemCompiler.reasoningContext(forProblem: problem)
    let generationContext = problemCompiler.generationContext(forProblem: problem)
    let perceivedInput = perceive(image: image, context: perceptionContext)
    let latent = reason(over: perceivedInput, context: reasoningContext)
    return generateImage(reasoningOutput: latent, context: generationContext)
  }

  // @differentiable
  public func generateImage(forNumber number: Tensor<Float>, problem: Problem) -> Tensor<Float> {
    let perceptionContext = problemCompiler.perceptionContext(forProblem: problem)
    let reasoningContext = problemCompiler.reasoningContext(forProblem: problem)
    let generationContext = problemCompiler.generationContext(forProblem: problem)
    let perceivedInput = perceive(number: number, context: perceptionContext)
    let latent = reason(over: perceivedInput, context: reasoningContext)
    return generateImage(reasoningOutput: latent, context: generationContext)
  }

  // @differentiable
  public func generateNumber(forImage image: Tensor<Float>, problem: Problem) -> Tensor<Float> {
    let perceptionContext = problemCompiler.perceptionContext(forProblem: problem)
    let reasoningContext = problemCompiler.reasoningContext(forProblem: problem)
    let generationContext = problemCompiler.generationContext(forProblem: problem)
    let perceivedInput = perceive(image: image, context: perceptionContext)
    let latent = reason(over: perceivedInput, context: reasoningContext)
    return generateNumber(reasoningOutput: latent, context: generationContext)
  }

  // @differentiable
  public func generateNumber(forNumber number: Tensor<Float>, problem: Problem) -> Tensor<Float> {
    let perceptionContext = problemCompiler.perceptionContext(forProblem: problem)
    let reasoningContext = problemCompiler.reasoningContext(forProblem: problem)
    let generationContext = problemCompiler.generationContext(forProblem: problem)
    let perceivedInput = perceive(number: number, context: perceptionContext)
    let latent = reason(over: perceivedInput, context: reasoningContext)
    return generateNumber(reasoningOutput: latent, context: generationContext)
  }
}

public struct ConvolutionalArchitecture: Architecture {
  @noDerivative public let hiddenSize: Int

  public var problemCompiler: LinearProblemCompiler
  public var numberEmbeddings: Tensor<Float>
  public var percConv1: ContextualizedLayer<Conv2D<Float>, Dense<Float>>
  public var percPool1: MaxPool2D<Float>
  public var percConv2: ContextualizedLayer<Conv2D<Float>, Dense<Float>>
  public var percPool2: MaxPool2D<Float>
  public var percFlatten: Flatten<Float>
  public var percFC1: ContextualizedLayer<Dense<Float>, Dense<Float>>
  public var percFC2: ContextualizedLayer<Dense<Float>, Dense<Float>>
  public var genFC1: ContextualizedLayer<Dense<Float>, Dense<Float>>
  public var genFC2: ContextualizedLayer<Dense<Float>, Dense<Float>>
  public var genReshape: Reshape<Float>
  public var genTransposedConv1: ContextualizedLayer<TransposedConv2D<Float>, Dense<Float>>
  public var genTransposedConv2: ContextualizedLayer<TransposedConv2D<Float>, Dense<Float>>
  public var reasoningLayer: ContextualizedLayer<Dense<Float>, Dense<Float>>

  public init(
    hiddenSize: Int,
    problemCompiler: LinearProblemCompiler,
    initializerStandardDeviation: Float = 0.02
  ) {
    let initializer = truncatedNormalInitializer(
      standardDeviation: Tensor<Float>(initializerStandardDeviation))
    self.hiddenSize = hiddenSize
    self.problemCompiler = problemCompiler
    self.numberEmbeddings = initializer([10, hiddenSize])
    let percConv1Base = Conv2D<Float>(filterShape: (5, 5, 3, 32), padding: .same, activation: gelu)
    self.percConv1 = ContextualizedLayer(
      base: percConv1Base,
      generator: Dense<Float>(
        inputSize: problemCompiler.problemEmbeddingSize,
        outputSize: percConv1Base.parameterCount,
        weightInitializer: initializer))
    self.percPool1 = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    let percConv2Base = Conv2D<Float>(filterShape: (5, 5, 32, 64), padding: .same, activation: gelu)
    self.percConv2 = ContextualizedLayer(
      base: percConv2Base,
      generator: Dense<Float>(
        inputSize: problemCompiler.problemEmbeddingSize,
        outputSize: percConv2Base.parameterCount,
        weightInitializer: initializer))
    self.percPool2 = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    self.percFlatten = Flatten<Float>()
    let percFC1Base = Dense<Float>(inputSize: 7 * 7 * 64, outputSize: 1024, activation: gelu)
    self.percFC1 = ContextualizedLayer(
      base: percFC1Base,
      generator: Dense<Float>(
        inputSize: problemCompiler.problemEmbeddingSize,
        outputSize: percFC1Base.parameterCount,
        weightInitializer: initializer))
    let percFC2Base = Dense<Float>(inputSize: 1024, outputSize: hiddenSize)
    self.percFC2 = ContextualizedLayer(
      base: percFC2Base,
      generator: Dense<Float>(
        inputSize: problemCompiler.problemEmbeddingSize,
        outputSize: percFC2Base.parameterCount,
        weightInitializer: initializer))
    let genFC1Base = Dense<Float>(inputSize: hiddenSize, outputSize: 1024, activation: gelu)
    self.genFC1 = ContextualizedLayer(
      base: genFC1Base,
      generator: Dense<Float>(
        inputSize: problemCompiler.problemEmbeddingSize,
        outputSize: genFC1Base.parameterCount,
        weightInitializer: initializer))
    let genFC2Base = Dense<Float>(inputSize: 1024, outputSize: 7 * 7 * 64, activation: gelu)
    self.genFC2 = ContextualizedLayer(
      base: genFC2Base,
      generator: Dense<Float>(
        inputSize: problemCompiler.problemEmbeddingSize,
        outputSize: genFC2Base.parameterCount,
        weightInitializer: initializer))
    self.genReshape = Reshape<Float>(shape: [-1, 7, 7, 64])
    let genTransposedConv1Base = TransposedConv2D<Float>(
      filterShape: (5, 5, 32, 64),
      strides: (2, 2),
      padding: .same,
      activation: gelu)
    self.genTransposedConv1 = ContextualizedLayer(
      base: genTransposedConv1Base,
      generator: Dense<Float>(
        inputSize: problemCompiler.problemEmbeddingSize,
        outputSize: genTransposedConv1Base.parameterCount,
        weightInitializer: initializer))
    let genTransposedConv2Base = TransposedConv2D<Float>(
      filterShape: (5, 5, 3, 32),
      strides: (2, 2),
      padding: .same)
    self.genTransposedConv2 = ContextualizedLayer(
      base: genTransposedConv2Base,
      generator: Dense<Float>(
        inputSize: problemCompiler.problemEmbeddingSize,
        outputSize: genTransposedConv2Base.parameterCount,
        weightInitializer: initializer))
    let reasoningLayerBase = Dense<Float>(
      inputSize: hiddenSize,
      outputSize: hiddenSize,
      weightInitializer: initializer)
    self.reasoningLayer = ContextualizedLayer(
      base: reasoningLayerBase,
      generator: Dense<Float>(
        inputSize: problemCompiler.problemEmbeddingSize,
        outputSize: reasoningLayerBase.parameterCount,
        weightInitializer: initializer))
  }

  /// - Returns: Tensor with shape `[batchSize, hiddenSize]`.
  @differentiable
  public func perceive(image: Tensor<Float>, context: Tensor<Float>) -> Tensor<Float> {
    let t1 = percPool1(percConv1(ContextualizedInput(input: image, context: context)))
    let t2 = percPool2(percConv2(ContextualizedInput(input: t1, context: context)))
    let t3 = percFlatten(t2)
    let t4 = percFC1(ContextualizedInput(input: t3, context: context))
    return percFC2(ContextualizedInput(input: t4, context: context))
  }

  /// - Returns: Tensor with shape `[batchSize, hiddenSize]`.
  @differentiable
  public func perceive(number: Tensor<Float>, context: Tensor<Float>) -> Tensor<Float> {
    let indices = withoutDerivative(at: number, in: Tensor<Int32>.init)
    return numberEmbeddings.gathering(atIndices: indices)
  }

  @differentiable
  public func reason(over input: Tensor<Float>, context: Tensor<Float>) -> Tensor<Float> {
    reasoningLayer(ContextualizedInput(input: input, context: context))
  }

  @differentiable
  public func generateImage(
    reasoningOutput: Tensor<Float>,
    context: Tensor<Float>
  ) -> Tensor<Float> {
    let t1 = genFC1(ContextualizedInput(input: reasoningOutput, context: context))
    let t2 = genFC2(ContextualizedInput(input: t1, context: context))
    let t3 = genReshape(t2)
    let t4 = genTransposedConv1(ContextualizedInput(input: t3, context: context))
    let t5 = genTransposedConv2(ContextualizedInput(input: t4, context: context))
    return sigmoid(t5)
  }

  @differentiable
  public func generateNumber(
    reasoningOutput: Tensor<Float>,
    context: Tensor<Float>
   ) -> Tensor<Float> {
    matmul(reasoningOutput, transposed: false, numberEmbeddings, transposed: true)
  }
}

extension ConvolutionalArchitecture {
  @differentiable
  public func generateImage(forImage image: Tensor<Float>, problem: Problem) -> Tensor<Float> {
    let perceptionContext = problemCompiler.perceptionContext(forProblem: problem)
    let reasoningContext = problemCompiler.reasoningContext(forProblem: problem)
    let generationContext = problemCompiler.generationContext(forProblem: problem)
    let perceivedInput = perceive(image: image, context: perceptionContext)
    let latent = reason(over: perceivedInput, context: reasoningContext)
    return generateImage(reasoningOutput: latent, context: generationContext)
  }

  @differentiable
  public func generateImage(forNumber number: Tensor<Float>, problem: Problem) -> Tensor<Float> {
    let perceptionContext = problemCompiler.perceptionContext(forProblem: problem)
    let reasoningContext = problemCompiler.reasoningContext(forProblem: problem)
    let generationContext = problemCompiler.generationContext(forProblem: problem)
    let perceivedInput = perceive(number: number, context: perceptionContext)
    let latent = reason(over: perceivedInput, context: reasoningContext)
    return generateImage(reasoningOutput: latent, context: generationContext)
  }

  @differentiable
  public func generateNumber(forImage image: Tensor<Float>, problem: Problem) -> Tensor<Float> {
    let perceptionContext = problemCompiler.perceptionContext(forProblem: problem)
    let reasoningContext = problemCompiler.reasoningContext(forProblem: problem)
    let generationContext = problemCompiler.generationContext(forProblem: problem)
    let perceivedInput = perceive(image: image, context: perceptionContext)
    let latent = reason(over: perceivedInput, context: reasoningContext)
    return generateNumber(reasoningOutput: latent, context: generationContext)
  }

  @differentiable
  public func generateNumber(forNumber number: Tensor<Float>, problem: Problem) -> Tensor<Float> {
    let perceptionContext = problemCompiler.perceptionContext(forProblem: problem)
    let reasoningContext = problemCompiler.reasoningContext(forProblem: problem)
    let generationContext = problemCompiler.generationContext(forProblem: problem)
    let perceivedInput = perceive(number: number, context: perceptionContext)
    let latent = reason(over: perceivedInput, context: reasoningContext)
    return generateNumber(reasoningOutput: latent, context: generationContext)
  }
}
