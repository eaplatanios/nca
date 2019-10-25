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

/// A layer that applies layer normalization over a batch of inputs.
///
/// - Source: [Layer Normalization](https://arxiv.org/abs/1607.06450).
public struct LayerNormalization<Scalar: TensorFlowFloatingPoint>: Layer, Regularizable {
  /// Offset value, also known as beta.
  public var offset: Tensor<Scalar>

  /// Scale value, also known as gamma.
  public var scale: Tensor<Scalar>

  /// The axis along which normalization is performed.
  @noDerivative public let axis: Int

  /// The variance epsilon value.
  @noDerivative public let epsilon: Tensor<Scalar>

  public var regularizationValue: TangentVector {
    TangentVector(offset: Tensor(Scalar(0)), scale: Tensor(Scalar(0)))
  }

  /// Creates a layer normalization layer.
  public init(
    offset: Tensor<Scalar>,
    scale: Tensor<Scalar>,
    axis: Int,
    epsilon: Tensor<Scalar> = Tensor(1e-12)
  ) {
    self.offset = offset
    self.scale = scale
    self.axis = axis
    self.epsilon = epsilon
  }

  /// Creates a layer normalization layer.
  ///
  /// - Parameters:
  ///   - featureCount: The number of features.
  ///   - axis: The axis that should be normalized.
  ///   - epsilon: The small scalar added to variance.
  public init(featureCount: Int, axis: Int, epsilon: Tensor<Scalar> = Tensor(1e-12)) {
    self.init(
      offset: Tensor(zeros: [featureCount]),
      scale: Tensor(ones: [featureCount]),
      axis: axis,
      epsilon: epsilon)
  }

  /// Returns the output obtained from applying this layer to the given input.
  ///
  /// - Parameter input: Input to this layer.
  /// - Returns: Output of this layer.
  @differentiable
  public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    let positiveAxis = (input.rank + axis) % input.rank
    var broadcastShape = TensorShape(Array(repeating: 1, count: input.rank))
    broadcastShape[positiveAxis] = input.shape[positiveAxis]
    let offset = self.offset.reshaped(to: broadcastShape)
    let scale = self.scale.reshaped(to: broadcastShape)
    let moments = input.moments(alongAxes: positiveAxis)
    return (input - moments.mean) * rsqrt(moments.variance + epsilon) * scale + offset
  }
}
