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

import Core
import TensorFlow

public struct VerificationInput<Input, Output: Differentiable>: Differentiable {
  @noDerivative public let input: Input
  public var output: Output

  @differentiable
  public init(input: Input, output: Output) {
    self.input = input
    self.output = output
  }
}

public typealias ModelInput = Tensor<Float>
public typealias ModelOutput = Tensor<Float>

public protocol Verifier: Layer
where Input == VerificationInput<ModelInput, ModelOutput>,
      Output == Tensor<Float> {
  // associatedtype ModelInput
  // associatedtype ModelOutput

  func initialInferenceModelOutput(forInput input: ModelInput) -> ModelOutput
  
  @differentiable
  func callAsFunction(_ input: VerificationInput<ModelInput, ModelOutput>) -> Tensor<Float>

  func callAsFunction<InfereceOptimizer: Core.Optimizer>(
    outputFor input: ModelInput,
    optimizer: InfereceOptimizer
  ) -> ModelOutput where InfereceOptimizer.Model == ModelOutput
}

fileprivate func projectToSimplex(_ values: Tensor<Float>) -> Tensor<Float> {
  let k = Int32(values.shape[values.rank - 1])
  let (u, _) = _Raw.topKV2(values, k: Tensor<Int32>(k), sorted: true)
  let cssv = u.cumulativeSum(alongAxis: -1)
  let rho = Tensor<Int32>(
    u * Tensor<Float>(rangeFrom: 1, to: Float(k) + 1, stride: 1) .> cssv - 1
  ).sum(alongAxes: -1)
  let theta = (cssv.batchGathering(atIndices: rho - 1) - 1) / Tensor<Float>(rho)
  return max(values - theta, 0)
}

extension Verifier {
  public func callAsFunction<InfereceOptimizer: Core.Optimizer>(
    outputFor input: ModelInput,
    optimizer: InfereceOptimizer
  ) -> ModelOutput where InfereceOptimizer.Model == ModelOutput {
    var output = initialInferenceModelOutput(forInput: input)
    var optimizer = optimizer
    for _ in 0..<10 {
      let (_, gradient) = TensorFlow.valueWithGradient(at: output) { output -> Tensor<Float> in
        let verificationInput = VerificationInput(input: input, output: output)
        return self(verificationInput).mean()
      }
      optimizer.update(&output, along: gradient)
      output = projectToSimplex(output)
    }
    return output
  }

  public mutating func train<TrainingOptimizer: Core.Optimizer, InferenceOptimizer: Core.Optimizer>(
    using dataset: Dataset,
    trainingOptimizer: TrainingOptimizer,
    inferenceOptimizer: InferenceOptimizer
  ) where TrainingOptimizer.Model == Self, InferenceOptimizer.Model == ModelOutput {
    var dataIterator = dataset.partitions[.train]!
      .makeIterator()
      .repeated()
      .shuffled(bufferSize: 1000)
      .map { index -> Example in
        let srcNumber = dataset.numbers[index]
        let tgtNumber = srcNumber
        let input = dataset.images[index]
        let output = Tensor<Float>(oneHotAtIndices: Tensor<Int32>(Int32(tgtNumber)), depth: 10)
        return Example(input: input, output: output)
      }
      .batched(batchSize: batchSize)
      .prefetched(count: 2)
    var trainingOptimizer = trainingOptimizer
    var accumulatedLoss = Float(0)
    var accumulatedSteps = 0
    for step in 0..<1000000 {
      let batch = dataIterator.next()!
      accumulatedSteps += 1
      accumulatedLoss += withLearningPhase(.training) {
        let (loss, gradient) = TensorFlow.valueWithGradient(at: self) { verifier -> Tensor<Float> in
          let positive = verifier(VerificationInput(input: batch.input, output: batch.output))
          let negative = verifier(VerificationInput(input: batch.input, output: Tensor<Float>(randomUniform: batch.output.shape)))
          return (negative - positive).mean()
        }
        trainingOptimizer.update(&self, along: gradient)
        return loss.scalarized()
      }
      if step % 10 == 0 {
        let loss = accumulatedLoss / Float(accumulatedSteps)
        print("Step: \(step), Loss: \(loss), Evaluation: \(evaluate(using: dataset, optimizer: inferenceOptimizer))")
        accumulatedLoss = 0
        accumulatedSteps = 0
      }
    }
  }

  public func evaluate<InferenceOptimizer: Core.Optimizer>(
    using dataset: Dataset,
    optimizer: InferenceOptimizer
  ) -> String where InferenceOptimizer.Model == ModelOutput {
    func exampleMap(_ index: Int) -> Example {
      let input = dataset.images[index]
      let output = Tensor<Float>(dataset.numbers[index])
      return Example(input: input, output: output)
    }

    return withLearningPhase(.inference) {
      var dataIterator = dataset.partitions[.test]!
        .makeIterator()
        .map(exampleMap)
        .batched(batchSize: 1000)
        .prefetched(count: 2)
      var correctCount = 0
      var totalCount = 0
      while let batch = dataIterator.next() {
        let predictions = self(
          outputFor: batch.input,
          optimizer: optimizer
        ).argmax(squeezingAxis: -1)
        let correct = Tensor<Int32>(predictions .== Tensor<Int32>(batch.output))
        correctCount += Int(correct.sum().scalarized())
        totalCount += predictions.shape[0]
      }
      return "acccuracy \(Float(correctCount) / Float(totalCount))"
    }
  }
}

public struct LeNetVerifier: Verifier {
  public var conv1 = Conv2D<Float>(filterShape: (5, 5, 3, 32), padding: .same, activation: gelu)
  public var pool1 = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
  @noDerivative public let dropout1 = Dropout<Float>(probability: 0.2)
  public var conv2 = Conv2D<Float>(filterShape: (5, 5, 32, 64), padding: .same, activation: gelu)
  public var pool2 = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
  @noDerivative public let dropout2 = Dropout<Float>(probability: 0.2)
  public var conv3 = Conv2D<Float>(filterShape: (3, 3, 64, 128), padding: .same, activation: gelu)
  public var pool3 = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
  @noDerivative public let dropout3 = Dropout<Float>(probability: 0.2)
  public var conv4 = Conv2D<Float>(filterShape: (3, 3, 128, 128), padding: .same, activation: gelu)
  public var pool4 = MaxPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
  @noDerivative public let dropout4 = Dropout<Float>(probability: 0.2)
  public var flatten = Flatten<Float>()
  public var fc1 = Dense<Float>(inputSize: 10, outputSize: 128, activation: gelu)
  @noDerivative public let dropoutFc1 = Dropout<Float>(probability: 0.2)
  public var fcCombined = Dense<Float>(inputSize: 256, outputSize: 1024, activation: gelu)
  public var fcOutput = Dense<Float>(inputSize: 1024, outputSize: 1)

  public init() {}

  public func initialInferenceModelOutput(forInput input: ModelInput) -> ModelOutput {
    Tensor<Float>(repeating: 0.5, shape: [input.shape[0], 10])
  }

  @differentiable
  public func callAsFunction(_ input: VerificationInput<ModelInput, ModelOutput>) -> Tensor<Float> {
    let convolved1 = input.input.sequenced(through: conv1, pool1, dropout1, conv2, pool2, dropout2)
    let convolved2 = convolved1.sequenced(through: conv3, pool3, dropout3, conv4, pool4, dropout4)
    let image = flatten(convolved2)
    let labels = dropoutFc1(fc1(input.output))
    let combined = Tensor<Float>(concatenating: [image, labels], alongAxis: -1)
    return sigmoid(fcOutput(fcCombined(combined)))
  }
}
