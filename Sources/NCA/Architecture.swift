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
  associatedtype ProblemCompiler: NCA.ProblemCompiler
  associatedtype TextPerception: TextPerceptionModule

  var problemCompiler: ProblemCompiler { get set }
  var textPerception: TextPerception { get set }

  @differentiable
  func perceive(text: TextBatch) -> Tensor<Float>

  /// - Returns: Tensor with shape `[batchSize, hiddenSize]`.
  @differentiable
  func pool(text: TextBatch, perceivedText: Tensor<Float>, context: Tensor<Float>) -> Tensor<Float>

  @differentiable
  func reason(over input: Tensor<Float>, context: Tensor<Float>) -> Tensor<Float>

  @differentiable
  func score(reasoningOutput: Tensor<Float>, concepts: [Concept]) -> Tensor<Float>
}

extension Architecture {
  public func preprocess(sequences: [String], maxSequenceLength: Int?) -> TextBatch {
    textPerception.preprocess(sequences: sequences, maxSequenceLength: maxSequenceLength)
  }

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
    problem: Problem,
    concepts: [Concept]
  ) -> Tensor<Float> {
// TODO: !!! Swift compiler AutoDiff bug.
//    var latent = Tensor<Float>(zeros: [])
//    if let text = input.text {
//      latent += pool(
//        text: text,
//        perceivedText: perceive(text: text),
//        problem: problem)
//    }
    let context = problemCompiler.compile(problem: problem)
    var latent = pool(
      text: input.text!,
      perceivedText: perceive(text: input.text!),
      context: context)
    latent = reason(over: latent, context: context)
    return classify(reasoningOutput: latent, concepts: concepts)
  }

  @differentiable
  public func score(
    _ input: ArchitectureInput,
    problem: Problem,
    concept: Concept
  ) -> Tensor<Float> {
// TODO: !!! Swift compiler AutoDiff bug.
//    var latent = Tensor<Float>(zeros: [])
//    if let text = input.text {
//      latent += pool(
//        text: text,
//        perceivedText: perceive(text: text),
//        problem: problem)
//    }
    let context = problemCompiler.compile(problem: problem)
    var latent = pool(
      text: input.text!,
      perceivedText: perceive(text: input.text!),
      context: context)
    latent = reason(over: latent, context: context)
    return score(reasoningOutput: latent, concepts: concept).squeezingShape(at: -1)
  }
}

public struct ArchitectureInput {
  public let text: TextBatch?

  public init(text: TextBatch? = nil) {
    self.text = text
  }
}

public struct SimpleArchitecture: Architecture {
  @noDerivative public let hiddenSize: Int
  @noDerivative public let reasoningHiddenSize: Int

  public var problemCompiler: SimpleProblemCompiler
  public var textPerception: BERT
  public var textPoolingQueryDense: Affine<Float>
  public var textPoolingMultiHeadAttention: MultiHeadAttention
  public var textPoolingOutputDense: ContextualizedLayer<Affine<Float>, Linear<Float>>
  public var reasoning: ContextualizedLayer<Sequential<Affine<Float>, Affine<Float>>, Linear<Float>>
  public var reasoningLayerNormalization: LayerNormalization<Float>
  public var reasoningToConceptDense: Affine<Float>

  public var regularizationValue: TangentVector {
    TangentVector(
      problemCompiler: problemCompiler.regularizationValue,
      textPerception: textPerception.regularizationValue,
      textPoolingQueryDense: textPoolingQueryDense.regularizationValue,
      textPoolingMultiHeadAttention: textPoolingMultiHeadAttention.regularizationValue,
      textPoolingOutputDense: textPoolingOutputDense.regularizationValue,
      reasoning: reasoning.regularizationValue,
      reasoningLayerNormalization: reasoningLayerNormalization.regularizationValue,
      reasoningToConceptDense: reasoningToConceptDense.regularizationValue)
  }

  public init(
    problemCompiler: SimpleProblemCompiler,
    textPerception: BERT,
    hiddenSize: Int,
    reasoningHiddenSize: Int
  ) {
    self.problemCompiler = problemCompiler
    self.textPerception = textPerception
    self.hiddenSize = hiddenSize
    self.reasoningHiddenSize = reasoningHiddenSize
    self.textPoolingQueryDense = Affine<Float>(
      inputSize: problemCompiler.problemEmbeddingSize,
      outputSize: textPerception.hiddenSize,
      weightInitializer: truncatedNormalInitializer(
        standardDeviation: Tensor(textPerception.initializerStandardDeviation)))
    self.textPoolingMultiHeadAttention = MultiHeadAttention(
      sourceSize: textPerception.hiddenSize,
      targetSize: textPerception.hiddenSize,
      headCount: textPerception.attentionHeadCount,
      headSize: textPerception.hiddenSize / textPerception.attentionHeadCount,
      queryActivation: { $0 },
      keyActivation: { $0 },
      valueActivation: { $0 },
      attentionDropoutProbability: textPerception.attentionDropoutProbability,
      matrixResult: true)
    let textPoolingOutputDenseBase = Affine<Float>(
      inputSize: textPerception.hiddenSize,
      outputSize: hiddenSize,
      weightInitializer: truncatedNormalInitializer(
        standardDeviation: Tensor(textPerception.initializerStandardDeviation)))
    self.textPoolingOutputDense = ContextualizedLayer(
      base: textPoolingOutputDenseBase,
      generator: Linear<Float>(
        inputSize: problemCompiler.problemEmbeddingSize,
        outputSize: textPoolingOutputDenseBase.parameterCount,
        weightInitializer: truncatedNormalInitializer(
          standardDeviation: Tensor(textPerception.initializerStandardDeviation))))
    let reasoningBase = Sequential(
      Affine<Float>(
        inputSize: hiddenSize,
        outputSize: reasoningHiddenSize,
        activation: textPerception.intermediateActivation,
        weightInitializer: truncatedNormalInitializer(
          standardDeviation: Tensor(textPerception.initializerStandardDeviation))),
      Affine<Float>(
        inputSize: reasoningHiddenSize,
        outputSize: hiddenSize,
        activation: textPerception.intermediateActivation,
        weightInitializer: truncatedNormalInitializer(
          standardDeviation: Tensor(textPerception.initializerStandardDeviation))))
    self.reasoning = ContextualizedLayer(
      base: reasoningBase,
      generator: Linear<Float>(
        inputSize: problemCompiler.problemEmbeddingSize,
        outputSize: reasoningBase.parameterCount,
        weightInitializer: truncatedNormalInitializer(
          standardDeviation: Tensor(textPerception.initializerStandardDeviation))))
    self.reasoningLayerNormalization = LayerNormalization<Float>(
      featureCount: hiddenSize,
      axis: -1)
    self.reasoningToConceptDense = Affine<Float>(
      inputSize: hiddenSize,
      outputSize: problemCompiler.conceptEmbeddingSize,
      weightInitializer: truncatedNormalInitializer(
        standardDeviation: Tensor(textPerception.initializerStandardDeviation)))
  }

  @differentiable
  public func perceive(text: TextBatch) -> Tensor<Float> {
    textPerception(text)
  }

  @differentiable
  public func pool(
    text: TextBatch,
    perceivedText: Tensor<Float>,
    context: Tensor<Float>
  ) -> Tensor<Float> {
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
  public func reason(over input: Tensor<Float>, context: Tensor<Float>) -> Tensor<Float> {
    let contextualizedInput = ContextualizedInput(input: input, context: context)
    return reasoningLayerNormalization(input + reasoning(contextualizedInput))
  }

  @differentiable
  public func score(reasoningOutput: Tensor<Float>, concepts: [Concept]) -> Tensor<Float> {
    let reasoning = reasoningToConceptDense(reasoningOutput)
    let compiledConcepts = Tensor<Float>(
      concatenating: problemCompiler.compile(concepts: concepts),
      alongAxis: 0)
    return matmul(reasoning, transposed: false, compiledConcepts, transposed: true)
  }
}
