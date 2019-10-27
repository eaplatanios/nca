// Copyright 2019, Emmanouil Antonios Platanios. All Rights Reserved.
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

public protocol Architecture: KeyPathIterable, Regularizable
where TangentVector: KeyPathIterable {
  @differentiable
  func perceive(text: TextBatch, problem: Problem) -> Tensor<Float>

  /// - Returns: Tensor with shape `[batchSize, hiddenSize]`.
  @differentiable
  func pool(text: TextBatch, perceivedText: Tensor<Float>, problem: Problem) -> Tensor<Float>

  @differentiable
  func reason(over input: Tensor<Float>, problem: Problem) -> Tensor<Float>

  @differentiable
  func classify(reasoningOutput: Tensor<Float>, problem: Classification) -> Tensor<Float>

  @differentiable
  func label(reasoningOutput: Tensor<Float>, problem: Labeling) -> Tensor<Float>
}

extension Architecture {
  @differentiable
  public func classify(_ input: ArchitectureInput, problem: Classification) -> Tensor<Float> {
// TODO: !!! Swift compiler AutoDiff bug.
//    var latent = Tensor<Float>(zeros: [])
//    if let text = input.text {
//      latent += pool(
//        text: text,
//        perceivedText: perceive(text: text, problem: problem),
//        problem: problem)
//    }
    var latent = pool(
      text: input.text!,
      perceivedText: perceive(text: input.text!, problem: problem),
      problem: problem)
    latent = reason(over: latent, problem: problem)
    return classify(reasoningOutput: latent, problem: problem)
  }
}

public struct ArchitectureInput {
  public let text: TextBatch?

  public init(text: TextBatch? = nil) {
    self.text = text
  }
}

public enum Context: Int, CaseIterable {
  case grammaticalCorrectness = 0
  case paraphrasing = 1
  case entailment = 2
  case sentiment = 3
}

public enum Concept: Int, CaseIterable {
  case positive = 0
  case negative = 1
  case neutral = 2
}

public protocol Problem {
  var context: Context { get }
}

public struct Classification: Problem {
  public let context: Context
  public let concepts: [Concept]

  public init(context: Context, concepts: [Concept]) {
    self.context = context
    self.concepts = concepts
  }
}

public struct Labeling: Problem {
  public let context: Context
  public let concepts: [Concept]

  public init(context: Context, concepts: [Concept]) {
    self.context = context
    self.concepts = concepts
  }
}

public struct SimpleArchitecture: Architecture {
  @noDerivative public let contextEmbeddingSize: Int
  @noDerivative public let hiddenSize: Int
  @noDerivative public var step: UInt64

  public var contextEmbeddings: Tensor<Float>
  public var conceptEmbeddings: Tensor<Float>
  public var textPerception: ALBERT
  public var textPoolingQueryDense: Affine<Float>
  public var textPoolingMultiHeadAttention: MultiHeadAttention
  public var textPoolingOutputDense: ContextualizedLayer<Affine<Float>, Linear<Float>>
  public var reasoning: ContextualizedLayer<Sequential<Affine<Float>, Affine<Float>>, Linear<Float>>
  public var reasoningLayerNormalization: LayerNormalization<Float>

  public var regularizationValue: TangentVector {
    TangentVector(
      contextEmbeddings: contextEmbeddings,
      conceptEmbeddings: conceptEmbeddings,
      textPerception: textPerception.regularizationValue,
      textPoolingQueryDense: textPoolingQueryDense.regularizationValue,
      textPoolingMultiHeadAttention: textPoolingMultiHeadAttention.regularizationValue,
      textPoolingOutputDense: textPoolingOutputDense.regularizationValue,
      reasoning: reasoning.regularizationValue,
      reasoningLayerNormalization: reasoningLayerNormalization.regularizationValue)
  }

  public init(
    albertConfiguration: ALBERT.Configuration,
    hiddenSize: Int,
    contextEmbeddingSize: Int,
    reasoningHiddenSize: Int,
    step: UInt64 = 0
  ) {
    self.hiddenSize = hiddenSize
    self.contextEmbeddingSize = contextEmbeddingSize
    self.step = step
    let initializer = truncatedNormalInitializer(
      standardDeviation: Tensor<Float>(albertConfiguration.initializerStandardDeviation))
    self.contextEmbeddings = initializer([Context.allCases.count, contextEmbeddingSize])
    self.conceptEmbeddings = initializer([Concept.allCases.count, hiddenSize])
    self.textPerception = ALBERT(configuration: albertConfiguration)
    self.textPoolingQueryDense = Affine<Float>(
      inputSize: contextEmbeddingSize,
      outputSize: albertConfiguration.hiddenSize,
      weightInitializer: truncatedNormalInitializer(
        standardDeviation: Tensor(albertConfiguration.initializerStandardDeviation)))
    self.textPoolingMultiHeadAttention = MultiHeadAttention(
      sourceSize: albertConfiguration.hiddenSize,
      targetSize: albertConfiguration.hiddenSize,
      headCount: albertConfiguration.attentionHeadCount,
      headSize: albertConfiguration.hiddenSize / albertConfiguration.attentionHeadCount,
      queryActivation: { $0 },
      keyActivation: { $0 },
      valueActivation: { $0 },
      attentionDropoutProbability: albertConfiguration.attentionDropoutProbability,
      matrixResult: true)
    let textPoolingOutputDenseBase = Affine<Float>(
      inputSize: albertConfiguration.hiddenSize,
      outputSize: hiddenSize,
      weightInitializer: truncatedNormalInitializer(
        standardDeviation: Tensor(albertConfiguration.initializerStandardDeviation)))
    self.textPoolingOutputDense = ContextualizedLayer(
      base: textPoolingOutputDenseBase,
      generator: Linear<Float>(
        inputSize: contextEmbeddingSize,
        outputSize: textPoolingOutputDenseBase.parameterCount,
        weightInitializer: truncatedNormalInitializer(
          standardDeviation: Tensor(albertConfiguration.initializerStandardDeviation))))
    let reasoningBase = Sequential(
      Affine<Float>(
        inputSize: hiddenSize,
        outputSize: reasoningHiddenSize,
        activation: albertConfiguration.intermediateActivation.activationFunction(),
        weightInitializer: truncatedNormalInitializer(
          standardDeviation: Tensor(albertConfiguration.initializerStandardDeviation))),
      Affine<Float>(
        inputSize: reasoningHiddenSize,
        outputSize: hiddenSize,
        activation: albertConfiguration.intermediateActivation.activationFunction(),
        weightInitializer: truncatedNormalInitializer(
          standardDeviation: Tensor(albertConfiguration.initializerStandardDeviation))))
    self.reasoning = ContextualizedLayer(
      base: reasoningBase,
      generator: Linear<Float>(
        inputSize: contextEmbeddingSize,
        outputSize: reasoningBase.parameterCount,
        weightInitializer: truncatedNormalInitializer(
          standardDeviation: Tensor(albertConfiguration.initializerStandardDeviation))))
    self.reasoningLayerNormalization = LayerNormalization<Float>(
      featureCount: hiddenSize,
      axis: -1)
  }

  @differentiable
  public func perceive(text: TextBatch, problem: Problem) -> Tensor<Float> {
    textPerception(text)
  }

  @differentiable
  public func pool(
    text: TextBatch,
    perceivedText: Tensor<Float>,
    problem: Problem
  ) -> Tensor<Float> {
    let context = contextEmbeddings[problem.context.rawValue].expandingShape(at: 0)
    let query = textPoolingQueryDense(context)
      .expandingShape(at: 0)
      .tiled(multiples: Tensor([Int32(perceivedText.shape[0]), 1, 1]))
    let attentionInput = AttentionInput(
      source: query,
      target: perceivedText,
      mask: Tensor<Float>(text.mask.expandingShape(at: 1)))
    let pooledPerceivedText = textPoolingMultiHeadAttention(attentionInput)
    return textPoolingOutputDense(ContextualizedInput(
      input: pooledPerceivedText,
      context: context))
  }

  @differentiable
  public func reason(over input: Tensor<Float>, problem: Problem) -> Tensor<Float> {
    let context = contextEmbeddings[problem.context.rawValue].expandingShape(at: 0)
    let contextualizedInput = ContextualizedInput(input: input, context: context)
    // We are adding a skip connection here to help the training process.
    return reasoningLayerNormalization(input + reasoning(contextualizedInput))
  }

  @differentiable
  public func classify(reasoningOutput: Tensor<Float>, problem: Classification) -> Tensor<Float> {
    let conceptIds = withoutDerivative(at: problem.concepts) {
      Tensor($0.map { Int32($0.rawValue) })
    }
    let classes = conceptEmbeddings.gathering(atIndices: conceptIds)
    let logits = matmul(reasoningOutput, transposed: false, classes, transposed: true)
    return logSoftmax(logits)
  }

  @differentiable
  public func label(reasoningOutput: Tensor<Float>, problem: Labeling) -> Tensor<Float> {
    let conceptIds = withoutDerivative(at: problem.concepts) {
      Tensor($0.map { Int32($0.rawValue) })
    }
    let classes = conceptEmbeddings.gathering(atIndices: conceptIds)
    let logits = matmul(reasoningOutput, transposed: false, classes, transposed: true)
    return logSigmoid(logits)
  }
}
