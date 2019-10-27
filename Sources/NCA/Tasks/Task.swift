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

public protocol Task {
  associatedtype EvaluationResult: Result

  mutating func update<A: Architecture, O: Optimizer>(
    architecture: inout A,
    using optimizer: inout O
  ) -> Float where O.Model == A

  func evaluate<A: Architecture>(using architecture: A) -> EvaluationResult
}

public protocol Result {
  var summary: String { get }
}
