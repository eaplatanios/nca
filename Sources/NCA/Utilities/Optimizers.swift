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

/// Represents a type that can contribute to the regularization term when training models.
public protocol Regularizable: Differentiable {
  /// The contribution of this term to the regularization term. This should be set to
  /// `TangentVector.zero` if this term should not contribute to the regularization term
  /// (e.g., for layer normalization parameters).
  var regularizationValue: TangentVector { get }
}

/// A numerical optimizer.
///
/// Optimizers apply an optimization algorithm to update the differentiable models.
public protocol Optimizer {
  /// The type of the model whose parameters are optimized.
  associatedtype Model: Differentiable

  /// Updates the provided model along the specified direction.
  mutating func update(_ model: inout Model, along direction: Model.TangentVector)
}

/// Adam optimizer.
///
/// Reference: ["Adam - A Method for Stochastic Optimization"](
/// https://arxiv.org/abs/1412.6980v8)
public struct Adam<Model: Differentiable, LearningRate: ScheduledParameter>: Optimizer
where Model.TangentVector: VectorProtocol & PointwiseMultiplicative &
                           ElementaryFunctions & KeyPathIterable,
      Model.TangentVector.VectorSpaceScalar == Float,
      LearningRate.Scalar == Float {
  /// The learning rate to use when updating models.
  public var learningRate: LearningRate

  /// A coefficient used to calculate the first and second moments of the gradients.
  public var beta1: Float

  /// A coefficient used to calculate the first and second moments of the gradients.
  public var beta2: Float

  /// A small scalar added to the denominator to improve numerical stability.
  public var epsilon: Float

  /// The maximum allowed gradient global norm. If the gradients global norm is larger than this
  /// value, then the gradients will be clipped to satisfy this constraint.
  public var maxGradientGlobalNorm: Float?

  /// The current step.
  public var step: UInt64 = 0

  /// The first moments of the weights.
  public var firstMoments: Model.TangentVector = .zero

  /// The second moments of the weights.
  public var secondMoments: Model.TangentVector = .zero

  public init(
    for model: __shared Model,
    learningRate: LearningRate,
    beta1: Float = 0.9,
    beta2: Float = 0.999,
    epsilon: Float = 1e-6,
    maxGradientGlobalNorm: Float? = nil
  ) {
    precondition(0 <= beta1 && beta1 <= 1, "Beta parameter must be between 0 and 1")
    precondition(0 <= beta2 && beta2 <= 1, "Beta parameter must be between 0 and 1")

    self.learningRate = learningRate
    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon
    self.maxGradientGlobalNorm = maxGradientGlobalNorm
  }

  public mutating func update(_ model: inout Model, along direction: Model.TangentVector) {
    var direction = direction
    if let globalNorm = maxGradientGlobalNorm {
      direction.clipByGlobalNorm(clipNorm: globalNorm)
    }
    step += 1
    firstMoments = firstMoments.scaled(by: beta1)
    firstMoments += direction.scaled(by: 1 - beta1)
    secondMoments = secondMoments.scaled(by: beta2)
    secondMoments += direction .* direction.scaled(by: 1 - beta2)
    let denominator = Model.TangentVector.sqrt(secondMoments).adding(epsilon)
    let update = firstMoments ./ denominator
    let step = Float(self.step)
    var learningRate = self.learningRate(forStep: self.step)
    learningRate *= sqrt(1 - pow(beta2, step)) / (1 - pow(beta1, step))
    model.move(along: update.scaled(by: -learningRate))
  }
}

/// Adam optimizer with weight decay.
///
/// Reference: ["Adam - A Method for Stochastic Optimization"](
/// https://arxiv.org/abs/1412.6980v8)
public struct WeightDecayedAdam<Model: Regularizable, LearningRate: ScheduledParameter>: Optimizer
where Model.TangentVector: VectorProtocol & PointwiseMultiplicative &
                           ElementaryFunctions & KeyPathIterable,
      Model.TangentVector.VectorSpaceScalar == Float,
      LearningRate.Scalar == Float {
  /// The learning rate to use when updating models.
  public var learningRate: LearningRate

  /// The weight decay rate.
  public var weightDecayRate: Float

  /// An indicator for whether or not to use bias correction.
  public var useBiasCorrection: Bool

  /// A coefficient used to calculate the first and second moments of the gradients.
  public var beta1: Float

  /// A coefficient used to calculate the first and second moments of the gradients.
  public var beta2: Float

  /// A small scalar added to the denominator to improve numerical stability.
  public var epsilon: Float

  /// The maximum allowed gradient global norm. If the gradients global norm is larger than this
  /// value, then the gradients will be clipped to satisfy this constraint.
  public var maxGradientGlobalNorm: Float?

  /// The current step.
  public var step: UInt64 = 0

  /// The first moments of the weights.
  public var firstMoments: Model.TangentVector = .zero

  /// The second moments of the weights.
  public var secondMoments: Model.TangentVector = .zero

  public init(
    for model: __shared Model,
    learningRate: LearningRate,
    weightDecayRate: Float = 0.01,
    useBiasCorrection: Bool = true,
    beta1: Float = 0.9,
    beta2: Float = 0.999,
    epsilon: Float = 1e-6,
    maxGradientGlobalNorm: Float? = nil
  ) {
    precondition(0 <= beta1 && beta1 <= 1, "Beta parameter must be between 0 and 1")
    precondition(0 <= beta2 && beta2 <= 1, "Beta parameter must be between 0 and 1")

    self.learningRate = learningRate
    self.weightDecayRate = weightDecayRate
    self.useBiasCorrection = useBiasCorrection
    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon
    self.maxGradientGlobalNorm = maxGradientGlobalNorm
  }

  public mutating func update(_ model: inout Model, along direction: Model.TangentVector) {
    var direction = direction
    if let globalNorm = maxGradientGlobalNorm {
      direction.clipByGlobalNorm(clipNorm: globalNorm)
    }
    step += 1
    firstMoments = firstMoments.scaled(by: beta1)
    firstMoments += direction.scaled(by: 1 - beta1)
    secondMoments = secondMoments.scaled(by: beta2)
    secondMoments += direction .* direction.scaled(by: 1 - beta2)
    let denominator = Model.TangentVector.sqrt(secondMoments).adding(epsilon)
    let weightDecay = model.regularizationValue.scaled(by: weightDecayRate)
    let update = firstMoments ./ denominator + weightDecay
    var learningRate = self.learningRate(forStep: step)
    if useBiasCorrection {
      let step = Float(self.step)
      learningRate *= sqrt(1 - pow(beta2, step)) / (1 - pow(beta1, step))
    }
    model.move(along: update.scaled(by: -learningRate))
  }
}

/// AMSGrad optimizer.
///
/// This algorithm is a modification of Adam with better convergence properties when close to local
/// optima.
///
/// Reference: ["On the Convergence of Adam and Beyond"](
/// https://openreview.net/pdf?id=ryQu7f-RZ)
public struct AMSGrad<
  Model: Differentiable & KeyPathIterable,
  LearningRate: ScheduledParameter
>: Optimizer where Model.TangentVector: VectorProtocol & PointwiseMultiplicative &
                                        ElementaryFunctions & KeyPathIterable,
                   Model.TangentVector.VectorSpaceScalar == Float,
                   LearningRate.Scalar == Float {
  /// The learning rate to use when updating models.
  public var learningRate: LearningRate

  /// A coefficient used to calculate the first and second moments of the gradients.
  public var beta1: Float

  /// A coefficient used to calculate the first and second moments of the gradients.
  public var beta2: Float

  /// A small scalar added to the denominator to improve numerical stability.
  public var epsilon: Float

  /// The maximum allowed gradient global norm. If the gradients global norm is larger than this
  /// value, then the gradients will be clipped to satisfy this constraint.
  public var maxGradientGlobalNorm: Float?

  /// The current step.
  public var step: UInt64 = 0

  /// The first moments of the weights.
  public var firstMoments: Model.TangentVector = .zero

  /// The second moments of the weights.
  public var secondMoments: Model.TangentVector = .zero

  /// The maximum of the second moments of the weights.
  public var secondMomentsMax: Model.TangentVector = .zero

  public init(
    for model: __shared Model,
    learningRate: LearningRate,
    beta1: Float = 0.9,
    beta2: Float = 0.999,
    epsilon: Float = 1e-6,
    maxGradientGlobalNorm: Float? = nil
  ) {
    precondition(0 <= beta1 && beta1 <= 1, "Beta parameter must be between 0 and 1")
    precondition(0 <= beta2 && beta2 <= 1, "Beta parameter must be between 0 and 1")

    self.learningRate = learningRate
    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon
    self.maxGradientGlobalNorm = maxGradientGlobalNorm
  }

  public mutating func update(_ model: inout Model, along direction: Model.TangentVector) {
    var direction = direction
    if let globalNorm = maxGradientGlobalNorm {
      direction.clipByGlobalNorm(clipNorm: globalNorm)
    }
    step += 1
    firstMoments = firstMoments.scaled(by: beta1)
    firstMoments += direction.scaled(by: 1 - beta1)
    secondMoments = secondMoments.scaled(by: beta2)
    secondMoments += direction .* direction.scaled(by: 1 - beta2)

    // Update `secondMomentsMax` using a key path approach because `max(_:_:)` cannot be
    // currently applied in a simpler manner.
    if step == 1 { secondMomentsMax = secondMoments }
    for kp in secondMomentsMax.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
      secondMomentsMax[keyPath: kp] = max(
        secondMomentsMax[keyPath: kp], secondMoments[keyPath: kp])
    }
    for kp in secondMomentsMax.recursivelyAllWritableKeyPaths(to: Tensor<Double>.self) {
      secondMomentsMax[keyPath: kp] = max(
        secondMomentsMax[keyPath: kp], secondMoments[keyPath: kp])
    }

    let denominator = Model.TangentVector.sqrt(secondMomentsMax).adding(epsilon)
    let update = firstMoments ./ denominator
    let step = Float(self.step)
    var learningRate = self.learningRate(forStep: self.step)
    learningRate *= sqrt(1 - pow(beta2, step)) / (1 - pow(beta1, step))
    model.move(along: update.scaled(by: -learningRate))
  }
}

/// LAMB optimizer.
///
/// Reference: ["Large Batch Optimization for Deep Learning"](
///              https://arxiv.org/abs/1904.00962)
public struct LAMB<
  Model: Regularizable & KeyPathIterable,
  LearningRate: ScheduledParameter
>: Optimizer where Model.TangentVector: VectorProtocol & PointwiseMultiplicative &
                                        ElementaryFunctions & KeyPathIterable,
                   Model.TangentVector.VectorSpaceScalar == Float,
                   LearningRate.Scalar == Float {
  /// The learning rate to use when updating models.
  public var learningRate: LearningRate

  /// The weight decay rate.
  public var weightDecayRate: Float

  /// An indicator for whether or not to use bias correction.
  public var useBiasCorrection: Bool

  /// A coefficient used to calculate the first and second moments of the gradients.
  public var beta1: Float

  /// A coefficient used to calculate the first and second moments of the gradients.
  public var beta2: Float

  /// A small scalar added to the denominator to improve numerical stability.
  public var epsilon: Float

  /// The maximum allowed gradient global norm. If the gradients global norm is larger than this
  /// value, then the gradients will be clipped to satisfy this constraint.
  public var maxGradientGlobalNorm: Float?

  /// The current step.
  public var step: UInt64 = 0

  /// The first moments of the weights.
  public var firstMoments: Model.TangentVector = .zero

  /// The second moments of the weights.
  public var secondMoments: Model.TangentVector = .zero

  public init(
    for model: __shared Model,
    learningRate: LearningRate,
    weightDecayRate: Float = 0.01,
    useBiasCorrection: Bool = true,
    beta1: Float = 0.9,
    beta2: Float = 0.999,
    epsilon: Float = 1e-6,
    maxGradientGlobalNorm: Float? = nil
  ) {
    precondition(0 <= beta1 && beta1 <= 1, "Beta parameter must be between 0 and 1")
    precondition(0 <= beta2 && beta2 <= 1, "Beta parameter must be between 0 and 1")

    self.learningRate = learningRate
    self.weightDecayRate = weightDecayRate
    self.useBiasCorrection = useBiasCorrection
    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon
    self.maxGradientGlobalNorm = maxGradientGlobalNorm
  }

  public mutating func update(_ model: inout Model, along direction: Model.TangentVector) {
    var direction = direction
    if let globalNorm = maxGradientGlobalNorm {
      direction.clipByGlobalNorm(clipNorm: globalNorm)
    }
    step += 1
    firstMoments = firstMoments.scaled(by: beta1)
    firstMoments += direction.scaled(by: 1 - beta1)
    secondMoments = secondMoments.scaled(by: beta2)
    secondMoments += direction .* direction.scaled(by: 1 - beta2)
    let denominator = Model.TangentVector.sqrt(secondMoments).adding(epsilon)
    let weightDecay = model.regularizationValue.scaled(by: weightDecayRate)
    let update = firstMoments ./ denominator + weightDecay
    var learningRate = self.learningRate(forStep: step)
    if useBiasCorrection {
      let step = Float(self.step)
      learningRate *= sqrt(1 - pow(beta2, step)) / (1 - pow(beta1, step))
    }
    for (kp, modelKp) in zip(
      update.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self),
      model.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self)
    ) {
      let r1 = sqrt((model[keyPath: modelKp].squared()).sum()).scalarized()
      let r2 = sqrt((update[keyPath: kp].squared()).sum()).scalarized()
      let r = r1 > 0 && r2 > 0 ? r1 / r2 : 1
      model[keyPath: modelKp].move(along: update[keyPath: kp].scaled(by: -learningRate * r))
    }
    for (kp, modelKp) in zip(
      update.recursivelyAllWritableKeyPaths(to: Tensor<Double>.self),
      model.recursivelyAllWritableKeyPaths(to: Tensor<Double>.self)
    ) {
      let r1 = sqrt((model[keyPath: modelKp].squared()).sum()).scalarized()
      let r2 = sqrt((update[keyPath: kp].squared()).sum()).scalarized()
      let r = r1 > 0 && r2 > 0 ? r1 / r2 : 1
      model[keyPath: modelKp].move(along: update[keyPath: kp].scaled(by: -learningRate * Float(r)))
    }
  }
}
