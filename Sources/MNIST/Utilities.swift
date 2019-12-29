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

import Logging
import TensorFlow

internal let logger = Logger(label: "NCA-MNIST")

/// Returns a function that creates a tensor by initializing all its values randomly from a
/// truncated Normal distribution. The generated values follow a Normal distribution with mean
/// `mean` and standard deviation `standardDeviation`, except that values whose magnitude is more
/// than two standard deviations from the mean are dropped and resampled.
///
/// - Parameters:
///   - mean: Mean of the Normal distribution.
///   - standardDeviation: Standard deviation of the Normal distribution.
///
///- Returns: A truncated normal parameter initializer function.
public func truncatedNormalInitializer<Scalar: TensorFlowFloatingPoint>(
  mean: Tensor<Scalar> = Tensor<Scalar>(0),
  standardDeviation: Tensor<Scalar> = Tensor<Scalar>(1),
  seed: TensorFlowSeed = TensorFlow.Context.local.randomSeed
) -> ParameterInitializer<Scalar> {
  {
    Tensor<Scalar>(
      randomTruncatedNormal: $0,
      mean: mean,
      standardDeviation: standardDeviation,
      seed: seed)
  }
}

//===------------------------------------------------------------------------------------------===//
// Protocols
//===------------------------------------------------------------------------------------------===//

extension KeyPathIterable {
  public mutating func clipByGlobalNorm<Scalar: TensorFlowFloatingPoint>(clipNorm: Scalar) {
    let clipNorm = Tensor<Scalar>(clipNorm)
    var globalNorm = Tensor<Scalar>(zeros: [])
    for kp in self.recursivelyAllWritableKeyPaths(to: Tensor<Scalar>.self) {
      globalNorm += self[keyPath: kp].squared().sum()
    }
    globalNorm = sqrt(globalNorm)
    for kp in self.recursivelyAllWritableKeyPaths(to: Tensor<Scalar>.self) {
      self[keyPath: kp] *= clipNorm / max(globalNorm, clipNorm)
    }
  }
}

public protocol Batchable {
  static func batch(_ values: [Self]) -> Self
}

extension Tensor: Batchable {
  public static func batch(_ values: [Tensor]) -> Tensor {
    Tensor(stacking: values, alongAxis: 0)
  }
}

extension KeyPathIterable {
  public static func batch(_ values: [Self]) -> Self {
    var result = values[0]
    for kp in result.recursivelyAllWritableKeyPaths(to: Tensor<UInt8>.self) {
      result[keyPath: kp] = Tensor.batch(values.map { $0[keyPath: kp] })
    }
    for kp in result.recursivelyAllWritableKeyPaths(to: Tensor<Int32>.self) {
      result[keyPath: kp] = Tensor.batch(values.map { $0[keyPath: kp] })
    }
    for kp in result.recursivelyAllWritableKeyPaths(to: Tensor<Int64>.self) {
      result[keyPath: kp] = Tensor.batch(values.map { $0[keyPath: kp] })
    }
    for kp in result.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
      result[keyPath: kp] = Tensor.batch(values.map { $0[keyPath: kp] })
    }
    for kp in result.recursivelyAllWritableKeyPaths(to: Tensor<Double>.self) {
      result[keyPath: kp] = Tensor.batch(values.map { $0[keyPath: kp] })
    }
    for kp in result.recursivelyAllWritableKeyPaths(to: Tensor<UInt8>?.self) {
      let keyPathValues = values.map { $0[keyPath: kp] }
      if keyPathValues[0] != nil {
        result[keyPath: kp] = Tensor.batch(keyPathValues.map { $0! })
      } else {
        result[keyPath: kp] = nil
      }
    }
    for kp in result.recursivelyAllWritableKeyPaths(to: Tensor<Int32>?.self) {
      let keyPathValues = values.map { $0[keyPath: kp] }
      if keyPathValues[0] != nil {
        result[keyPath: kp] = Tensor.batch(keyPathValues.map { $0! })
      } else {
        result[keyPath: kp] = nil
      }
    }
    for kp in result.recursivelyAllWritableKeyPaths(to: Tensor<Int64>?.self) {
      let keyPathValues = values.map { $0[keyPath: kp] }
      if keyPathValues[0] != nil {
        result[keyPath: kp] = Tensor.batch(keyPathValues.map { $0! })
      } else {
        result[keyPath: kp] = nil
      }
    }
    for kp in result.recursivelyAllWritableKeyPaths(to: Tensor<Float>?.self) {
      let keyPathValues = values.map { $0[keyPath: kp] }
      if keyPathValues[0] != nil {
        result[keyPath: kp] = Tensor.batch(keyPathValues.map { $0! })
      } else {
        result[keyPath: kp] = nil
      }
    }
    for kp in result.recursivelyAllWritableKeyPaths(to: Tensor<Double>?.self) {
      let keyPathValues = values.map { $0[keyPath: kp] }
      if keyPathValues[0] != nil {
        result[keyPath: kp] = Tensor.batch(keyPathValues.map { $0! })
      } else {
        result[keyPath: kp] = nil
      }
    }
    return result
  }
}
