// Copyright 2020, Emmanouil Antonios Platanios. All Rights Reserved.
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

public extension Tensor {
  /// Returns a transposed tensor, with dimensions permuted in the specified order.
  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  func transposed(permutation: Tensor<Int32>) -> Tensor {
    _Raw.transpose(self, perm: permutation)
  }

  /// Returns a transposed tensor, with dimensions permuted in the specified order.
  @available(*, deprecated, renamed: "transposed(permutation:)")
  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  func transposed(withPermutations permutations: Tensor<Int32>) -> Tensor {
    transposed(permutation: permutations)
  }

  /// Returns a transposed tensor, with dimensions permuted in the specified order.
  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  func transposed(permutation: [Int]) -> Tensor {
    let permutation = permutation.map(Int32.init)
    return transposed(permutation: Tensor<Int32>(permutation))
  }

  /// Returns a transposed tensor, with dimensions permuted in the specified order.
  @available(*, deprecated, renamed: "transposed(permutation:)")
  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  func transposed(withPermutations permutations: [Int]) -> Tensor {
    transposed(permutation: permutations)
  }

  /// Returns a transposed tensor, with dimensions permuted in the specified order.
  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  func transposed(permutation: Int...) -> Tensor {
    transposed(permutation: permutation)
  }

  /// Returns a transposed tensor, with dimensions permuted in the specified order.
  @available(*, deprecated, renamed: "transposed(permutation:)")
  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  func transposed(withPermutations permutations: Int...) -> Tensor {
    transposed(permutation: permutations)
  }

  /// Returns a transposed tensor, with dimensions permuted in reverse order.
  @inlinable
  @differentiable(wrt: self where Scalar: TensorFlowFloatingPoint)
  func transposed() -> Tensor {
    let defaultPermutations = rankTensor - 1 - Tensor<Int32>(
      rangeFrom: 0, to: Int32(rank), stride: 1)
    return transposed(permutation: Tensor<Int32>(defaultPermutations))
  }
}

internal extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  @derivative(of: transposed(permutation:))
  func _vjpTransposed(permutation: Tensor<Int32>) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    let value = transposed(permutation: permutation)
    return (value, { $0.transposed(permutation: _Raw.invertPermutation(permutation)) })
  }
}
