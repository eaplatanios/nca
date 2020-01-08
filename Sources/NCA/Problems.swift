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

import Core
import TensorFlow

public enum Problem {
  case grammar
  case implication
  case equivalence
  case sentiment
}

// TODO: Add support for regularizers that try to figure out which problem the learner is currently
// tackling. E.g., predict that it's a grammar task by looking at CoLA sentence examples.
public enum Concept {
  case grammaticallyCorrect
  case grammaticallyIncorrect
  case implication
  case contradiction
  case neutral
  case equivalence
  case nonEquivalence
  case positiveSentiment
  case negativeSentiment
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

  public var problemEmbeddings: Tensor<Float>
  public var conceptEmbeddings: Tensor<Float>

  public var regularizationValue: TangentVector {
    TangentVector(
      problemEmbeddings: problemEmbeddings,
      conceptEmbeddings: conceptEmbeddings)
  }

  public init(
    problemEmbeddingSize: Int,
    conceptEmbeddingSize: Int,
    initializerStandardDeviation: Float = 0.02
  ) {
    self.problemEmbeddingSize = problemEmbeddingSize
    self.conceptEmbeddingSize = conceptEmbeddingSize
    let initializer = truncatedNormalInitializer(
      standardDeviation: Tensor<Float>(initializerStandardDeviation))
    self.problemEmbeddings = initializer([4, problemEmbeddingSize])
    self.conceptEmbeddings = initializer([9, conceptEmbeddingSize])
  }

  @differentiable
  public func compile(problem: Problem) -> Tensor<Float> {
    switch problem {
    case .grammar: return problemEmbeddings[0].expandingShape(at: 0)
    case .implication: return problemEmbeddings[1].expandingShape(at: 0)
    case .equivalence: return problemEmbeddings[2].expandingShape(at: 0)
    case .sentiment: return problemEmbeddings[3].expandingShape(at: 0)
    }
  }

  @differentiable
  public func compile(concept: Concept) -> Tensor<Float> {
    switch concept {
    case .grammaticallyCorrect: return conceptEmbeddings[0].expandingShape(at: 0)
    case .grammaticallyIncorrect: return conceptEmbeddings[1].expandingShape(at: 0)
    case .implication: return conceptEmbeddings[2].expandingShape(at: 0)
    case .contradiction: return conceptEmbeddings[3].expandingShape(at: 0)
    case .neutral: return conceptEmbeddings[4].expandingShape(at: 0)
    case .equivalence: return conceptEmbeddings[5].expandingShape(at: 0)
    case .nonEquivalence: return conceptEmbeddings[6].expandingShape(at: 0)
    case .positiveSentiment: return conceptEmbeddings[7].expandingShape(at: 0)
    case .negativeSentiment: return conceptEmbeddings[8].expandingShape(at: 0)
    }
  }
}
