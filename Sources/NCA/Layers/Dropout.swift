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

/// Dropout layer.
///
/// Dropout consists of randomly zeroing out a fraction of the input units during training time.
/// This helps prevent overfitting.
public struct Dropout<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
  @noDerivative public let probability: Scalar

  /// Creates a dropout layer.
  ///
  /// - Parameter probability: The drop probability.
  public init(probability: Scalar) {
    self.probability = probability
  }

  /// Returns the output obtained from applying this layer to the given input.
  ///
  /// - Parameter input: Input to this layer.
  /// - Returns: Output of this layer.
  @differentiable
  public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    switch TensorFlow.Context.local.learningPhase {
    case .training: return input.droppingOut(probability: probability)
    case .inference: return input
    }
  }
}
