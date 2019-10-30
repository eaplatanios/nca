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
  func perceive(text: TextBatch) -> Tensor<Float>

  /// - Returns: Tensor with shape `[batchSize, hiddenSize]`.
  @differentiable
  func pool(text: TextBatch, perceivedText: Tensor<Float>, problem: Problem) -> Tensor<Float>

  @differentiable
  func reason(over input: Tensor<Float>, problem: Problem) -> Tensor<Float>

  @differentiable
  func score(reasoningOutput: Tensor<Float>, concepts: [Concept]) -> Tensor<Float>
}

extension Architecture {
  @differentiable
  public func score(reasoningOutput: Tensor<Float>, concepts: Concept...) -> Tensor<Float> {
    score(reasoningOutput: reasoningOutput, concepts: concepts)
  }

  @differentiable
  public func classify(reasoningOutput: Tensor<Float>, concepts: [Concept]) -> Tensor<Float> {
    logSoftmax(score(reasoningOutput: reasoningOutput, concepts: concepts))
  }

  @differentiable
  public func label(reasoningOutput: Tensor<Float>, concepts: [Concept]) -> Tensor<Float> {
    logSigmoid(score(reasoningOutput: reasoningOutput, concepts: concepts))
  }

  @differentiable
  public func classify(
    _ input: ArchitectureInput,
    context: Context,
    concepts: [Concept]
  ) -> Tensor<Float> {
    let problem = Problem.scoring(context: context, concepts: concepts)
// TODO: !!! Swift compiler AutoDiff bug.
//    var latent = Tensor<Float>(zeros: [])
//    if let text = input.text {
//      latent += pool(
//        text: text,
//        perceivedText: perceive(text: text),
//        problem: problem)
//    }
    var latent = pool(
      text: input.text!,
      perceivedText: perceive(text: input.text!),
      problem: problem)
    latent = reason(over: latent, problem: problem)
    return classify(reasoningOutput: latent, concepts: concepts)
  }

  @differentiable
  public func score(
    _ input: ArchitectureInput,
    context: Context,
    concept: Concept
  ) -> Tensor<Float> {
    let problem = Problem.scoring(context: context, concepts: [concept])
// TODO: !!! Swift compiler AutoDiff bug.
//    var latent = Tensor<Float>(zeros: [])
//    if let text = input.text {
//      latent += pool(
//        text: text,
//        perceivedText: perceive(text: text),
//        problem: problem)
//    }
    var latent = pool(
      text: input.text!,
      perceivedText: perceive(text: input.text!),
      problem: problem)
    latent = reason(over: latent, problem: problem)
    return score(reasoningOutput: latent, concepts: concept).squeezingShape(at: -1)
  }
}

public struct ArchitectureInput {
  public let text: TextBatch?

  public init(text: TextBatch? = nil) {
    self.text = text
  }
}

public enum Context: Int, CaseIterable {
  case inputScoring = 0
}

public enum Concept: Int, CaseIterable {
  case grammaticalCorrectness = 0
  case entailment = 1
  case contradiction = 2
  case neutral = 3
  case positiveSentiment = 4
  case negativeSentiment = 5
  case paraphrasing = 6
  case equivalence = 7
}

public enum Problem {
  case scoring(context: Context, concepts: [Concept])
}

public struct SimpleArchitecture: Architecture {
  @noDerivative public let contextEmbeddingSize: Int
  @noDerivative public let conceptEmbeddingSize: Int
  @noDerivative public let hiddenSize: Int

  public var contextEmbeddings: Tensor<Float>
  public var conceptEmbeddings: Tensor<Float>
  public var conceptToContextDense: Affine<Float>
  public var contextConceptCombiner: Affine<Float>
  @Freezable public var textPerception: BERT
  public var textPoolingQueryDense: Affine<Float>
  public var textPoolingMultiHeadAttention: MultiHeadAttention
  public var textPoolingOutputDense: ContextualizedLayer<Affine<Float>, Linear<Float>>
  public var reasoning: ContextualizedLayer<Sequential<Affine<Float>, Affine<Float>>, Linear<Float>>
  public var reasoningLayerNormalization: LayerNormalization<Float>
  public var reasoningToConceptDense: Affine<Float>

  public var regularizationValue: TangentVector {
    TangentVector(
      contextEmbeddings: contextEmbeddings,
      conceptEmbeddings: conceptEmbeddings,
      conceptToContextDense: conceptToContextDense.regularizationValue,
      contextConceptCombiner: contextConceptCombiner.regularizationValue,
      _textPerception: textPerception.regularizationValue,
      textPoolingQueryDense: textPoolingQueryDense.regularizationValue,
      textPoolingMultiHeadAttention: textPoolingMultiHeadAttention.regularizationValue,
      textPoolingOutputDense: textPoolingOutputDense.regularizationValue,
      reasoning: reasoning.regularizationValue,
      reasoningLayerNormalization: reasoningLayerNormalization.regularizationValue,
      reasoningToConceptDense: reasoningToConceptDense.regularizationValue)
  }

  public init(
    bertConfiguration: BERT.Configuration,
    contextEmbeddingSize: Int,
    conceptEmbeddingSize: Int,
    hiddenSize: Int,
    reasoningHiddenSize: Int
  ) {
    self.contextEmbeddingSize = contextEmbeddingSize
    self.conceptEmbeddingSize = conceptEmbeddingSize
    self.hiddenSize = hiddenSize
    let initializer = truncatedNormalInitializer(
      standardDeviation: Tensor<Float>(bertConfiguration.initializerStandardDeviation))
    self.contextEmbeddings = initializer([Context.allCases.count, contextEmbeddingSize])
    self.conceptEmbeddings = initializer([Concept.allCases.count, conceptEmbeddingSize])
    self.conceptToContextDense = Affine<Float>(
      inputSize: conceptEmbeddingSize,
      outputSize: contextEmbeddingSize,
      weightInitializer: truncatedNormalInitializer(
        standardDeviation: Tensor(bertConfiguration.initializerStandardDeviation)))
    self.contextConceptCombiner = Affine<Float>(
      inputSize: 2 * contextEmbeddingSize,
      outputSize: contextEmbeddingSize,
      weightInitializer: truncatedNormalInitializer(
        standardDeviation: Tensor(bertConfiguration.initializerStandardDeviation)))
    self._textPerception = Freezable(wrappedValue: BERT(configuration: bertConfiguration))
    self.textPoolingQueryDense = Affine<Float>(
      inputSize: contextEmbeddingSize,
      outputSize: bertConfiguration.hiddenSize,
      weightInitializer: truncatedNormalInitializer(
        standardDeviation: Tensor(bertConfiguration.initializerStandardDeviation)))
    self.textPoolingMultiHeadAttention = MultiHeadAttention(
      sourceSize: bertConfiguration.hiddenSize,
      targetSize: bertConfiguration.hiddenSize,
      headCount: bertConfiguration.attentionHeadCount,
      headSize: bertConfiguration.hiddenSize / bertConfiguration.attentionHeadCount,
      queryActivation: { $0 },
      keyActivation: { $0 },
      valueActivation: { $0 },
      attentionDropoutProbability: bertConfiguration.attentionDropoutProbability,
      matrixResult: true)
    let textPoolingOutputDenseBase = Affine<Float>(
      inputSize: bertConfiguration.hiddenSize,
      outputSize: hiddenSize,
      weightInitializer: truncatedNormalInitializer(
        standardDeviation: Tensor(bertConfiguration.initializerStandardDeviation)))
    self.textPoolingOutputDense = ContextualizedLayer(
      base: textPoolingOutputDenseBase,
      generator: Linear<Float>(
        inputSize: contextEmbeddingSize,
        outputSize: textPoolingOutputDenseBase.parameterCount,
        weightInitializer: truncatedNormalInitializer(
          standardDeviation: Tensor(bertConfiguration.initializerStandardDeviation))))
    let reasoningBase = Sequential(
      Affine<Float>(
        inputSize: hiddenSize,
        outputSize: reasoningHiddenSize,
        activation: bertConfiguration.intermediateActivation.activationFunction(),
        weightInitializer: truncatedNormalInitializer(
          standardDeviation: Tensor(bertConfiguration.initializerStandardDeviation))),
      Affine<Float>(
        inputSize: reasoningHiddenSize,
        outputSize: hiddenSize,
        activation: bertConfiguration.intermediateActivation.activationFunction(),
        weightInitializer: truncatedNormalInitializer(
          standardDeviation: Tensor(bertConfiguration.initializerStandardDeviation))))
    self.reasoning = ContextualizedLayer(
      base: reasoningBase,
      generator: Linear<Float>(
        inputSize: contextEmbeddingSize,
        outputSize: reasoningBase.parameterCount,
        weightInitializer: truncatedNormalInitializer(
          standardDeviation: Tensor(bertConfiguration.initializerStandardDeviation))))
    self.reasoningLayerNormalization = LayerNormalization<Float>(
      featureCount: hiddenSize,
      axis: -1)
    self.reasoningToConceptDense = Affine<Float>(
      inputSize: hiddenSize,
      outputSize: conceptEmbeddingSize,
      weightInitializer: truncatedNormalInitializer(
        standardDeviation: Tensor(bertConfiguration.initializerStandardDeviation)))
  }

  public mutating func freezeTextPerception() {
    _textPerception.frozen = true
  }

  public mutating func unfreezeTextPerception() {
    _textPerception.frozen = false
  }

  @differentiable
  internal func context(for problem: Problem) -> Tensor<Float> {
    switch problem {
    case let .scoring(context, concepts):
      let context = contextEmbeddings[context.rawValue].expandingShape(at: 0)
      let conceptIds = withoutDerivative(at: concepts) { Tensor($0.map { Int32($0.rawValue) }) }
      let concepts = conceptEmbeddings.gathering(atIndices: conceptIds)
      let mappedConcepts = conceptToContextDense(concepts).sum(alongAxes: 0)
      let concatenated = Tensor(concatenating: [context, mappedConcepts], alongAxis: -1)
      return contextConceptCombiner(concatenated)
    }
  }

  @differentiable
  public func perceive(text: TextBatch) -> Tensor<Float> {
    textPerception(text)
  }

  @differentiable
  public func pool(
    text: TextBatch,
    perceivedText: Tensor<Float>,
    problem: Problem
  ) -> Tensor<Float> {
    let context = self.context(for: problem)
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
    let context = self.context(for: problem)
    let contextualizedInput = ContextualizedInput(input: input, context: context)
    return reasoningLayerNormalization(input + reasoning(contextualizedInput))
  }

  @differentiable
  public func score(reasoningOutput: Tensor<Float>, concepts: [Concept]) -> Tensor<Float> {
    let reasoning = reasoningToConceptDense(reasoningOutput)
    let conceptIds = withoutDerivative(at: concepts) { Tensor($0.map { Int32($0.rawValue) }) }
    let concepts = conceptEmbeddings.gathering(atIndices: conceptIds)
    return matmul(reasoning, transposed: false, concepts, transposed: true)
  }
}
