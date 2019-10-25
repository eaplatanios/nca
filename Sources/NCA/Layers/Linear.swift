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

/// A densely-connected neural network layer.
///
/// `Dense` implements the operation `activation(matmul(input, weight) + bias)`, where `weight` is
/// a weight matrix, `bias` is a bias vector, and `activation` is an element-wise activation
/// function.
///
/// This layer also supports 3-D weight tensors with 2-D bias matrices. In this case the first
/// dimension of both is treated as the batch size that is aligned with the first dimension of
/// `input` and the batch variant of the `matmul(_:_:)` operation is used, thus using a different
/// weight and bias for each element in input batch.
public struct Linear<Scalar: TensorFlowFloatingPoint>: Layer, Regularizable {
  /// The weight matrix.
  public var weight: Tensor<Scalar>

  /// The bias vector.
  public var bias: Tensor<Scalar>

  /// The element-wise activation function.
  @noDerivative public let activation: Activation<Scalar>

  public var regularizationValue: TangentVector {
    TangentVector(weight: weight, bias: Tensor(Scalar(0)))
  }

  /// Creates a linear layer with the specified input size, output size, and element-wise
  /// activation function. The weight matrix is created with shape `[inputSize, outputSize]` and
  /// the bias vector is created with shape `[outputSize]`.
  ///
  /// - Parameters:
  ///   - inputSize: The dimensionality of the input space.
  ///   - outputSize: The dimensionality of the output space.
  ///   - activation: The activation function to use. The default value is `identity(_:)`.
  ///   - weightInitializer: Initializer to use for `weight`.
  ///   - biasInitializer: Initializer to use for `bias`.
  init(
    inputSize: Int,
    outputSize: Int,
    activation: @escaping Activation<Scalar> = identity,
    weightInitializer: ParameterInitializer<Scalar> = glorotUniform(),
    biasInitializer: ParameterInitializer<Scalar> = zeros()
  ) {
    self.weight = weightInitializer([inputSize, outputSize])
    self.bias = biasInitializer([outputSize])
    self.activation = activation
  }

  /// Returns the output obtained from applying the layer to the given input.
  ///
  /// - Parameter input: The input to the layer.
  /// - Returns: The output.
  @differentiable
  public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    activation(matmul(input, weight) + bias)
  }
}
