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

/// Adam optimizer.
///
/// Reference: ["Adam - A Method for Stochastic Optimization"](
/// https://arxiv.org/abs/1412.6980v8)
public struct Adam<Model: Differentiable>: Optimizer
where Model.TangentVector: VectorProtocol & PointwiseMultiplicative & ElementaryFunctions,
      Model.TangentVector.VectorSpaceScalar == Float {
  /// The learning rate.
  public var learningRate: Float
  /// A coefficient used to calculate the first and second moments of the gradients.
  public var beta1: Float
  /// A coefficient used to calculate the first and second moments of the gradients.
  public var beta2: Float
  /// A small scalar added to the denominator to improve numerical stability.
  public var epsilon: Float
  /// The learning rate decay.
  public var decay: Float
  /// The current step.
  public var step: Int = 0
  /// The first moments of the weights.
  public var firstMoments: Model.TangentVector = .zero
  /// The second moments of the weights.
  public var secondMoments: Model.TangentVector = .zero

  public init(
    for model: __shared Model,
    learningRate: Float = 1e-3,
    beta1: Float = 0.9,
    beta2: Float = 0.999,
    epsilon: Float = 1e-8,
    decay: Float = 0
  ) {
    precondition(learningRate >= 0, "Learning rate must be non-negative")
    precondition(0 <= beta1 && beta1 <= 1, "Beta parameter must be between 0 and 1")
    precondition(0 <= beta2 && beta2 <= 1, "Beta parameter must be between 0 and 1")
    precondition(decay >= 0, "Learning rate decay must be non-negative")

    self.learningRate = learningRate
    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon
    self.decay = decay
  }

  public mutating func update(_ model: inout Model, along direction: Model.TangentVector) {
    step += 1
    let step = Float(self.step)
    let learningRate = self.learningRate * 1 / (1 + decay * step)
    // Note: `stepSize` and `secondMoments` are split into two lines to avoid the "compiler is
    // unable to type-check this expression in reasonable time" error.
    var stepSize = learningRate * sqrt(1 - pow(beta2, step))
    stepSize = stepSize / (1 - pow(beta1, step))
    firstMoments = firstMoments.scaled(by: beta1)
    firstMoments += direction.scaled(by: 1 - beta1)
    secondMoments = secondMoments.scaled(by: beta2)
    secondMoments += direction .* direction.scaled(by: 1 - beta2)
    let denominator = Model.TangentVector.sqrt(secondMoments).adding(epsilon)
    model.move(along: firstMoments.scaled(by: -stepSize) ./ denominator)
  }
}

/// AMSGrad optimizer.
///
/// This algorithm is a modification of Adam with better convergence properties when close to local
/// optima.
///
/// Reference: ["On the Convergence of Adam and Beyond"](
/// https://openreview.net/pdf?id=ryQu7f-RZ)
public struct AMSGrad<Model: Differentiable & KeyPathIterable>: Optimizer
where Model.TangentVector: VectorProtocol & PointwiseMultiplicative &
                           ElementaryFunctions & KeyPathIterable,
  Model.TangentVector.VectorSpaceScalar == Float {
  /// The learning rate.
  public var learningRate: Float
  /// A coefficient used to calculate the first and second moments of the gradients.
  public var beta1: Float
  /// A coefficient used to calculate the first and second moments of the gradients.
  public var beta2: Float
  /// A small scalar added to the denominator to improve numerical stability.
  public var epsilon: Float
  /// The learning rate decay.
  public var decay: Float
  /// The current step.
  public var step: Int = 0
  /// The first moments of the weights.
  public var firstMoments: Model.TangentVector = .zero
  /// The second moments of the weights.
  public var secondMoments: Model.TangentVector = .zero
  /// The maximum of the second moments of the weights.
  public var secondMomentsMax: Model.TangentVector = .zero

  public init(
    for model: __shared Model,
    learningRate: Float = 1e-3,
    beta1: Float = 0.9,
    beta2: Float = 0.999,
    epsilon: Float = 1e-8,
    decay: Float = 0
  ) {
    precondition(learningRate >= 0, "Learning rate must be non-negative")
    precondition(0 <= beta1 && beta1 <= 1, "Beta parameter must be between 0 and 1")
    precondition(0 <= beta2 && beta2 <= 1, "Beta parameter must be between 0 and 1")
    precondition(decay >= 0, "Learning rate decay must be non-negative")

    self.learningRate = learningRate
    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon
    self.decay = decay
  }

  public mutating func update(_ model: inout Model, along direction: Model.TangentVector) {
    step += 1
    let step = Float(self.step)
    let beta1Power = pow(beta1, step)
    let beta2Power = pow(beta2, step)
    let learningRate = self.learningRate * 1 / (1 + decay * step)
    // Note: `stepSize` and `secondMoments` are split into two lines to avoid the "compiler is
    // unable to type-check this expression in reasonable time" error.
    var stepSize = learningRate * sqrt(1 - pow(beta2Power, step))
    stepSize = stepSize / (1 - pow(beta1Power, step))
    firstMoments = firstMoments.scaled(by: beta1)
    firstMoments += direction.scaled(by: 1 - beta1)
    secondMoments = secondMoments.scaled(by: beta2)
    secondMoments += direction .* direction.scaled(by: 1 - beta2)

    // Update `secondMomentsMax` using a key path approach because `max(_:_:)` cannot be
    // currently applied in a simpler manner.
    for kp in secondMomentsMax.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
      secondMomentsMax[keyPath: kp] = max(
        secondMomentsMax[keyPath: kp], secondMoments[keyPath: kp])
    }
    for kp in secondMomentsMax.recursivelyAllWritableKeyPaths(to: Tensor<Double>.self) {
      secondMomentsMax[keyPath: kp] = max(
        secondMomentsMax[keyPath: kp], secondMoments[keyPath: kp])
    }

    let denominator = Model.TangentVector.sqrt(secondMomentsMax).adding(epsilon)
    model.move(along: firstMoments.scaled(by: -stepSize) ./ denominator)
  }
}
