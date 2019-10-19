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

extension Tensor {
  /// Returns this tensor reshaped to a matrix (i.e., a rank-2 tensor).
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  internal func reshapedToMatrix() -> Tensor {
    reshaped(to: [-1, shape[-1]])
  }

  /// Returns this previously-reshaped rank-2 tensor reshaped back to its original shape.
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  internal func reshapedFromMatrix(originalShape: TensorShape) -> Tensor {
    reshaped(to: TensorShape(
      originalShape[0..<originalShape.count - 1].dimensions + [shape[-1]]))
  }

  /// Returns this previously-reshaped rank-2 tensor reshaped back to its original shape.
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  internal func reshapedFromMatrix(originalShape: Tensor<Int32>) -> Tensor {
    reshaped(toShape: Tensor<Int32>(concatenating: [
      originalShape[0..<originalShape.shape[0] - 1],
      Tensor<Int32>([Int32(shape[-1])])
    ]))
  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  /// Returns this tensor with a random subset of its elements set to zero.
  ///
  /// - Parameters:
  ///   - probability: Probability that each element is zeroed out.
  ///   - scaleResult: If `true`, the result will be divided by the keep probability.
  ///
  /// - Precondition: `probability` must be in the interval `[0, 1)`.
  @differentiable(wrt: self)
  internal func droppingOut(probability: Scalar, scaleResult: Bool = true) -> Tensor {
    precondition(probability >= 0 && probability < 1, "The dropout probability must be in [0, 1).")
    if probability == Scalar(0) { return self }
    let keepProbability = Scalar(1.0 - probability)
    let random = Tensor(
      randomUniform: shape,
      lowerBound: Tensor<Scalar>(keepProbability),
      upperBound: Tensor<Scalar>(keepProbability + 1))
    let masked = self * floor(random)
    return scaleResult ? masked / Tensor(keepProbability) : masked
  }
}
