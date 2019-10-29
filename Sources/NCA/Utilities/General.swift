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

import CTensorFlow
import Foundation
import Logging
import TensorFlow

internal let logger = Logger(label: "NCA")

public typealias ParameterInitializer<Scalar: TensorFlowScalar> = (TensorShape) -> Tensor<Scalar>

public typealias Activation<Scalar: TensorFlowFloatingPoint> =
  @differentiable (Tensor<Scalar>) -> Tensor<Scalar>

public protocol Serializable {
  init(fromFile fileURL: URL) throws
  func save(toFile fileURL: URL) throws
}

extension Array {
  // TODO: [DOC] Add documentation.
  public func concurrentMap<B>(_ transform: @escaping (Element) -> B) -> [B] {
    var result = Array<B?>(repeating: nil, count: count)
    let queue = DispatchQueue(label: "concurrentMap Queue")
    DispatchQueue.concurrentPerform(iterations: count) { index in
      let element = self[index]
      let transformed = transform(element)
      queue.sync {
        result[index] = transformed
      }
    }
    return result.map { $0! }
  }
}
