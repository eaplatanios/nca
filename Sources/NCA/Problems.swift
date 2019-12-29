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

public enum Problem {
  case classify([Concept])
  case label([Concept])
}

// TODO: Add support for regularizers that try to figure out which problem the learner is currently
// tackling. E.g., predict that it's a grammar task by looking at CoLA sentence examples.
public enum Concept {
  case grammar
  case implication
  case equivalence
  case sentiment
  indirect case positive(Concept)
  indirect case negative(Concept)
  indirect case neutral(Concept)
}

public protocol ProblemCompiler: Regularizable {
  var problemEmbeddingSize: Int { get }
  var conceptEmbeddingSize: Int { get }

  @differentiable
  func compile(problem: Problem) -> Tensor<Float>

  @differentiable
  func compile(concept: Concept) -> Tensor<Float>
}

extension ProblemCompiler {
  @differentiable
  public func compile(concepts: [Concept]) -> [Tensor<Float>] {
    concepts.map(compile(concept:))
  }

  @usableFromInline
  @derivative(of: compile(concepts:), wrt: self)
  internal func _vjpCompile(concepts: [Concept]) -> (
    value: [Tensor<Float>],
    pullback: (Array<Tensor<Float>>.TangentVector) -> TangentVector
  ) {
    var values = [Tensor<Float>]()
    var pullbacks = [(Tensor<Float>) -> TangentVector]()
    values.reserveCapacity(concepts.count)
    pullbacks.reserveCapacity(concepts.count)
    for concept in concepts {
      let (value, pullback) = Swift.valueWithPullback(at: self) { $0.compile(concept: concept) }
      values.append(value)
      pullbacks.append(pullback)
    }
    return (values, { backpropagatedGradients -> TangentVector in
      var gradient = TangentVector.zero
      for (pullback, backpropagatedGradient) in zip(pullbacks, backpropagatedGradients) {
        gradient += pullback(backpropagatedGradient)
      }
      return gradient
    })
  }
}

public struct SimpleProblemCompiler: ProblemCompiler, KeyPathIterable {
  @noDerivative public let problemEmbeddingSize: Int
  @noDerivative public let conceptEmbeddingSize: Int
  @noDerivative public let modifierEmbeddingSize: Int
  @noDerivative public let conceptModifierHiddenSize: Int
  @noDerivative public let conceptModifierGeneratorHiddenSize: Int
  @noDerivative public let problemAttentionHeadCount: Int

  public var problemEmbeddings: Tensor<Float>
  public var conceptEmbeddings: Tensor<Float>
  public var modifierEmbeddings: Tensor<Float>
  public var problemAttention: MultiHeadAttention
  public var conceptModifier: ContextualizedLayer<
    Sequential<Affine<Float>, Affine<Float>>,
    Sequential<Affine<Float>, Affine<Float>>>

  public var regularizationValue: TangentVector {
    TangentVector(
      problemEmbeddings: problemEmbeddings,
      conceptEmbeddings: conceptEmbeddings,
      modifierEmbeddings: modifierEmbeddings,
      problemAttention: problemAttention.regularizationValue,
      conceptModifier: conceptModifier.regularizationValue)
  }

  public init(
    problemEmbeddingSize: Int,
    conceptEmbeddingSize: Int,
    modifierEmbeddingSize: Int,
    conceptModifierHiddenSize: Int,
    conceptModifierGeneratorHiddenSize: Int,
    problemAttentionHeadCount: Int,
    problemAttentionDropoutProbability: Float = 0.1,
    initializerStandardDeviation: Float = 0.02
  ) {
    self.problemEmbeddingSize = problemEmbeddingSize
    self.conceptEmbeddingSize = conceptEmbeddingSize
    self.modifierEmbeddingSize = modifierEmbeddingSize
    self.conceptModifierHiddenSize = conceptModifierHiddenSize
    self.conceptModifierGeneratorHiddenSize = conceptModifierGeneratorHiddenSize
    self.problemAttentionHeadCount = problemAttentionHeadCount
    let initializer = truncatedNormalInitializer(
      standardDeviation: Tensor<Float>(initializerStandardDeviation))
    self.problemEmbeddings = initializer([2, problemEmbeddingSize])
    self.conceptEmbeddings = initializer([4, conceptEmbeddingSize])
    self.modifierEmbeddings = initializer([3, modifierEmbeddingSize])
    self.problemAttention = MultiHeadAttention(
      sourceSize: problemEmbeddingSize,
      targetSize: conceptEmbeddingSize,
      headCount: problemAttentionHeadCount,
      headSize: problemEmbeddingSize / problemAttentionHeadCount,
      queryActivation: { $0 },
      keyActivation: { $0 },
      valueActivation: { $0 },
      attentionDropoutProbability: problemAttentionDropoutProbability,
      matrixResult: true)
    let conceptModifierBase = Sequential(
      Affine<Float>(
        inputSize: conceptEmbeddingSize,
        outputSize: conceptModifierHiddenSize,
        activation: gelu,
        weightInitializer: initializer),
      Affine<Float>(
        inputSize: conceptModifierHiddenSize,
        outputSize: conceptEmbeddingSize,
        weightInitializer: initializer))
    let conceptModifierGenerator = Sequential(
      Affine<Float>(
        inputSize: modifierEmbeddingSize,
        outputSize: conceptModifierGeneratorHiddenSize,
        activation: gelu,
        weightInitializer: initializer),
      Affine<Float>(
        inputSize: conceptModifierGeneratorHiddenSize,
        outputSize: conceptModifierBase.parameterCount,
        weightInitializer: initializer))
    self.conceptModifier = ContextualizedLayer(
      base: conceptModifierBase,
      generator: conceptModifierGenerator)
  }

  @differentiable
  public func compile(problem: Problem) -> Tensor<Float> {
    switch problem {
    case let .classify(concepts):
      let problemEmbedding = problemEmbeddings[0].expandingShape(at: 0, 1)
      let compiledConcepts = Tensor<Float>(
        concatenating: compile(concepts: concepts),
        alongAxis: 0
      ).expandingShape(at: 0)
      return problemAttention(AttentionInput(
        source: problemEmbedding,
        target: compiledConcepts,
        mask: Tensor<Float>(ones: [1, 1, compiledConcepts.shape[1]])))
    case let .label(concepts):
      let problemEmbedding = problemEmbeddings[1].expandingShape(at: 0, 1)
      let compiledConcepts = Tensor<Float>(
        concatenating: compile(concepts: concepts),
        alongAxis: 0
      ).expandingShape(at: 0)
      return problemAttention(AttentionInput(
        source: problemEmbedding,
        target: compiledConcepts,
        mask: Tensor<Float>(ones: [1, 1, compiledConcepts.shape[1]])))
    }
  }

  @differentiable
  public func compile(concept: Concept) -> Tensor<Float> {
    switch concept {
    case .grammar: return conceptEmbeddings[0].expandingShape(at: 0)
    case .implication: return conceptEmbeddings[1].expandingShape(at: 0)
    case .equivalence: return conceptEmbeddings[2].expandingShape(at: 0)
    case .sentiment: return conceptEmbeddings[3].expandingShape(at: 0)
    case let .positive(baseConcept):
      let concept = compile(concept: baseConcept)
      let modifier = modifierEmbeddings[0].expandingShape(at: 0)
      return conceptModifier(ContextualizedInput(input: concept, context: modifier))
    case let .negative(baseConcept):
      let concept = compile(concept: baseConcept)
      let modifier = modifierEmbeddings[1].expandingShape(at: 0)
      return conceptModifier(ContextualizedInput(input: concept, context: modifier))
    case let .neutral(baseConcept):
      let concept = compile(concept: baseConcept)
      let modifier = modifierEmbeddings[2].expandingShape(at: 0)
      return conceptModifier(ContextualizedInput(input: concept, context: modifier))
    }
  }
}
