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

public protocol Architecture: Differentiable {
  @differentiable
  func perceive(text: TextBatch) -> Tensor<Float>

  @differentiable
  func pool(text: Tensor<Float>) -> Tensor<Float>

  @differentiable
  func reason(over input: Tensor<Float>, problem: Problem) -> Tensor<Float>

  @differentiable
  func classify(reasoningOutput: Tensor<Float>, problem: Problem) -> Tensor<Float>

  @differentiable
  func label(reasoningOutput: Tensor<Float>, problem: Problem) -> Tensor<Float>
}

extension Architecture {
  @differentiable
  public func classify(_ input: ArchitectureInput, problem: Problem) -> Tensor<Float> {
    var latent = Tensor<Float>(zeros: [])
    if let text = input.text {
      latent += pool(text: perceive(text: text))
    }
    latent = reason(over: latent, problem: problem)
    return classify(reasoningOutput: latent, problem: problem)
  }
}

public struct ArchitectureInput {
  public let text: TextBatch?

  public init(text: TextBatch? = nil) {
    self.text = text
  }
}

public enum Context: Int, CaseIterable {
  case grammaticalCorrectness = 0
}

public enum Concept: Int, CaseIterable {
  case positive = 0
  case negative = 1
  case neutral = 2
}

public enum Problem {
  case classify(context: Context, concepts: [Concept])
  case label(context: Context, concepts: [Concept])
}
