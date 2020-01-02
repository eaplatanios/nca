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

    return withLearningPhase(.inference) {
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
}

public struct BatchedConv2D<Scalar: TensorFlowFloatingPoint>: Layer {
  public var filter: Tensor<Scalar>
  public var bias: Tensor<Scalar>
  @noDerivative public let activation: Activation
  @noDerivative public let strides: (Int, Int)
  @noDerivative public let padding: Padding
  @noDerivative public let dilations: (Int, Int)

  public typealias Activation = @differentiable (Tensor<Scalar>) -> Tensor<Scalar>

  public init(
    filter: Tensor<Scalar>,
    bias: Tensor<Scalar>,
    activation: @escaping Activation = identity,
    strides: (Int, Int) = (1, 1),
    padding: Padding = .valid,
    dilations: (Int, Int) = (1, 1)
  ) {
    self.filter = filter
    self.bias = bias
    self.activation = activation
    self.strides = strides
    self.padding = padding
    self.dilations = dilations
  }

  @differentiable
  public func callAsFunction(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    let filter = self.filter
      .transposed(permutation: 1, 2, 0, 3, 4)
      .reshaped(to: [
        self.filter.shape[1], self.filter.shape[2],
        self.filter.shape[0] * self.filter.shape[3], self.filter.shape[4]])
    let input = input
      .transposed(permutation: 1, 2, 0, 3)
      .reshaped(to: [1, input.shape[1], input.shape[2], input.shape[0] * input.shape[3]])
    let conv = depthwiseConv2D(
      input,
      filter: filter,
      strides:  (1, strides.0, strides.1, 1),
      padding: padding)
    // Please note that this (otherwise great) answer is wrong in its current form. If the padding = "VALID", than the out = tf.reshape(out, [H, W, MB, channels, out_channels) line should read out = tf.reshape(out, [H-fh+1, W-fw+1, MB, channels, out_channels) You form is correct, if you use padding = "SAME". See my answer bellow for the correct for, treating both cases.
    let result = conv
      .reshaped(to: [
        conv.shape[1], conv.shape[2],
        self.filter.shape[0], self.filter.shape[3],
        self.filter.shape[4]])
      .transposed(permutation: [2, 0, 1, 3, 4])
      .sum(squeezingAxes: 3)
    return activation(result + bias.expandingShape(at: 1, 2))
  }
}

public extension BatchedConv2D {
  init(
    filterShape: (Int, Int, Int, Int),
    strides: (Int, Int) = (1, 1),
    padding: Padding = .valid,
    activation: @escaping Activation = identity,
    filterInitializer: ParameterInitializer<Scalar> = glorotUniform(),
    biasInitializer: ParameterInitializer<Scalar> = zeros()
  ) {
    let filterTensorShape = TensorShape([
      filterShape.0, filterShape.1, filterShape.2, filterShape.3])
    self.init(
      filter: filterInitializer(filterTensorShape),
      bias: biasInitializer([filterShape.3]),
      activation: activation,
      strides: strides,
      padding: padding)
  }
}

public struct LeNet: Layer {
  public var conv1 = Conv2D<Float>(filterShape: (5, 5, 3, 32), padding: .same, activation: gelu)
  public var pool1 = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
  public var conv2 = Conv2D<Float>(filterShape: (5, 5, 32, 64), padding: .same, activation: gelu)
  public var pool2 = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
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
  public var norm1 = BatchNorm<Float>(featureCount: 3)
  public var conv1 = { () -> ContextualizedLayer<BatchedConv2D<Float>, Sequential<Conv2D<Float>, Sequential<Flatten<Float>, Dense<Float>>>> in
    let conv1Base = BatchedConv2D<Float>(
      filterShape: (5, 5, 3, 32),
      padding: .same,
      activation: gelu)
    let conv1Generator = Sequential {
      Conv2D<Float>(filterShape: (5, 5, 3, 32), strides: (2, 2), padding: .same, activation: gelu)
      Flatten<Float>()
      Dense<Float>(inputSize: 6272, outputSize: conv1Base.parameterCount)
    }
    return ContextualizedLayer(base: conv1Base, generator: conv1Generator)
  }()

  // public var conv1 = Conv2D<Float>(filterShape: (5, 5, 3, 32), padding: .same, activation: gelu)
  public var pool1 = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2))

  public var norm2 = BatchNorm<Float>(featureCount: 32)
  public var conv2 = { () -> ContextualizedLayer<BatchedConv2D<Float>, Sequential<Conv2D<Float>, Sequential<Flatten<Float>, Dense<Float>>>> in
    let conv2Base = BatchedConv2D<Float>(
      filterShape: (5, 5, 32, 64),
      padding: .same,
      activation: gelu)
    let conv2Generator = Sequential {
      Conv2D<Float>(filterShape: (5, 5, 32, 32), strides: (2, 2), padding: .same, activation: gelu)
      Flatten<Float>()
      Dense<Float>(inputSize: 1568, outputSize: conv2Base.parameterCount)
    }
    return ContextualizedLayer(base: conv2Base, generator: conv2Generator)
  }()

  // public var conv2 = Conv2D<Float>(filterShape: (5, 5, 32, 64), padding: .same, activation: gelu)
  public var pool2 = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
  public var flatten = Flatten<Float>()
  public var norm3 = BatchNorm<Float>(featureCount: 7 * 7 * 64)
  public var fc1 = Dense<Float>(inputSize: 7 * 7 * 64, outputSize: 1024, activation: gelu)
  public var fc2 = Dense<Float>(inputSize: 1024, outputSize: 10)

  public init() {}

  @differentiable
  public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
    let t1 = norm1(input)
    let t2 = norm2(pool1(conv1(ContextualizedInput(input: t1, context: t1))))
    let t3 = pool2(conv2(ContextualizedInput(input: t2, context: t2)))
    return fc2(fc1(norm3(flatten(t3))))
  }
}
