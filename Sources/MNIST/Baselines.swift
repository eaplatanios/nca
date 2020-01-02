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

extension IdentityTask {
  public mutating func update<L: Layer, O: Optimizer>(
    layer: inout L,
    using optimizer: inout O
  ) -> Float where O.Model == L, L.Input == Tensor<Float>, L.Output == Tensor<Float> {
    assert(srcModality == .image && tgtModality == .number)
    let batch = trnExamplesIterator.next()!
    return withLearningPhase(.training) {
      let (loss, gradient) = valueWithGradient(at: layer) {
        softmaxCrossEntropy(
          logits: $0(batch.input),
          labels: Tensor<Int32>(batch.output),
          reduction: { $0.mean() })
      }
      optimizer.update(&layer, along: gradient)
      return loss.scalarized()
    }
  }

  public func evaluate<L: Layer>(
    _ layer: L,
    using dataset: Dataset,
    batchSize: Int
  ) -> [String: Float] where L.Input == Tensor<Float>, L.Output == Tensor<Float> {
    assert(srcModality == .image && tgtModality == .number)

    func exampleMap(_ index: Int) -> Example {
      let input = dataset.images[index]
      let output = Tensor<Float>(dataset.numbers[index])
      return Example(input: input, output: output)
    }

    var tstExamples = dataset.partitions[.test]!
      .makeIterator()
      .map(exampleMap)
      .batched(batchSize: batchSize)
      .prefetched(count: 2)
    var correctCount = 0
    var totalCount = 0
    while let batch = tstExamples.next() {
      let predictions = Tensor<UInt8>(layer(batch.input).argmax(squeezingAxis: -1))
      let correct = Tensor<Int32>(predictions .== Tensor<UInt8>(batch.output))
      correctCount += Int(correct.sum().scalarized())
      totalCount += predictions.shape[0]
    }
    return ["acccuracy": Float(correctCount) / Float(totalCount)]
  }
}

public struct LeNet: Layer {
  public var conv1 = Conv2D<Float>(filterShape: (5, 5, 3, 32), padding: .same, activation: gelu)
  public var pool1 = AvgPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
  public var conv2 = Conv2D<Float>(filterShape: (5, 5, 32, 64), padding: .same, activation: gelu)
  public var pool2 = AvgPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
  public var flatten = Flatten<Float>()
  public var fc1 = Dense<Float>(inputSize: 7 * 7 * 64, outputSize: 1024, activation: gelu)
  public var fc2 = Dense<Float>(inputSize: 1024, outputSize: 10)

  public init() {}

  @differentiable
  public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
    let convolved = input.sequenced(through: conv1, pool1, conv2, pool2)
    return convolved.sequenced(through: flatten, fc1, fc2)
  }
}

public struct ContextualLeNet: Layer {
  public var conv1 = { () -> ContextualizedLayer<Conv2D<Float>, Sequential<Conv2D<Float>, Sequential<Flatten<Float>, Dense<Float>>>> in
    let conv1Base = Conv2D<Float>(filterShape: (5, 5, 3, 32), padding: .same, activation: gelu)
    let conv1Generator = Sequential {
      Conv2D<Float>(filterShape: (10, 10, 3, 8), strides: (5, 5), padding: .same, activation: gelu)
      Flatten<Float>()
      Dense<Float>(inputSize: 288, outputSize: conv1Base.parameterCount)
    }
    return ContextualizedLayer(base: conv1Base, generator: conv1Generator)
  }()

  // public var conv1 = Conv2D<Float>(filterShape: (5, 5, 3, 32), padding: .same, activation: gelu)
  public var pool1 = AvgPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
  public var conv2 = Conv2D<Float>(filterShape: (5, 5, 32, 64), padding: .same, activation: gelu)
  public var pool2 = AvgPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
  public var flatten = Flatten<Float>()
  public var fc1 = Dense<Float>(inputSize: 7 * 7 * 64, outputSize: 1024, activation: gelu)
  public var fc2 = Dense<Float>(inputSize: 1024, outputSize: 10)

  public init() {}

  @differentiable
  public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
    let t1 = ContextualizedInput(input: input, context: input)
    let t2 = pool1(conv1(t1))
    let convolved = t2.sequenced(through: conv2, pool2)
    return convolved.sequenced(through: flatten, fc1, fc2)
  }
}
