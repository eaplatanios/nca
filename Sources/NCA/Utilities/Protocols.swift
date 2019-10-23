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
    return result
  }
}
