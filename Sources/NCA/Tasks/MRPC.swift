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

import Foundation
import TensorFlow

public struct MRPC: Task {
  public let directoryURL: URL
  public let trainExamples: [Example]
  public let devExamples: [Example]
  public let testExamples: [Example]
  public let maxSequenceLength: Int
  public let batchSize: Int

  private typealias ExampleIterator = IndexingIterator<Array<Example>>
  private typealias RepeatExampleIterator = ShuffleIterator<RepeatIterator<ExampleIterator>>
  private typealias TrainDataIterator = PrefetchIterator<GroupedIterator<MapIterator<RepeatExampleIterator, DataBatch>>>
  private typealias DevDataIterator = PrefetchIterator<GroupedIterator<MapIterator<ExampleIterator, DataBatch>>>
  private typealias TestDataIterator = DevDataIterator

  private var trainDataIterator: TrainDataIterator
  private var devDataIterator: DevDataIterator
  private var testDataIterator: TestDataIterator

  public mutating func update<A: Architecture, O: Optimizer>(
    architecture: inout A,
    using optimizer: inout O
  ) -> Float where O.Model == A {
    let batch = withDevice(.cpu) { trainDataIterator.next()! }
    let input = ArchitectureInput(text: batch.inputs)
    let labels = Tensor<Float>(batch.labels)
    return withLearningPhase(.training) {
      let (loss, gradient) = architecture.valueWithGradient {
        sigmoidCrossEntropy(
          logits: $0.score(input, context: .inputScoring, concept: .paraphrasing),
          labels: labels,
          reduction: { $0.mean() })
      }
      optimizer.update(&architecture, along: gradient)
      return loss.scalarized()
    }
  }

  public func evaluate<A: Architecture>(using architecture: A) -> [String: Float] {
    var devDataIterator = self.devDataIterator.copy()
    var devPredictedLabels = [Bool]()
    var devGroundTruth = [Bool]()
    while let batch = withDevice(.cpu, perform: { devDataIterator.next() }) {
      let input = ArchitectureInput(text: batch.inputs)
      let predictions = architecture.score(
        input,
        context: .inputScoring,
        concept: .paraphrasing)
      let predictedLabels = sigmoid(predictions) .>= 0.5
      devPredictedLabels.append(contentsOf: predictedLabels.scalars)
      devGroundTruth.append(contentsOf: batch.labels.scalars.map { $0 == 1 })
    }
    return [
      "f1Score": NCA.f1Score(predictions: devPredictedLabels, groundTruth: devGroundTruth),
      "accuracy": NCA.accuracy(predictions: devPredictedLabels, groundTruth: devGroundTruth)]
  }
}

extension MRPC {
  public init<A: Architecture>(
    for architecture: A,
    taskDirectoryURL: URL,
    maxSequenceLength: Int,
    batchSize: Int
  ) throws {
    self.directoryURL = taskDirectoryURL.appendingPathComponent("MRPC")

    let dataURL = directoryURL.appendingPathComponent("data")
    let trainDataURL = dataURL.appendingPathComponent("msr_paraphrase_train.txt")
    let testDataURL = dataURL.appendingPathComponent("msr_paraphrase_test.txt")
    let devIdsURL = dataURL.appendingPathComponent("mrpc_dev_ids.tsv")

    // Download the data, if necessary.
    try maybeDownload(from: MRPC.trainDataURL, to: trainDataURL)
    try maybeDownload(from: MRPC.testDataURL, to: testDataURL)
    try maybeDownload(from: MRPC.devIdsURL, to: devIdsURL)

    // Load the dev IDs (which correspond to IDs in the train data file).
    let devIdsLines = try String(contentsOf: devIdsURL, encoding: .utf8).split { $0.isNewline }
    let devIds = devIdsLines.map { line -> String in
      let parts = line.split(separator: "\t")
        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
      return "\(parts[0]):\(parts[1])"
    }

    // Load the train and dev examples.
    var trainExamples = [Example]()
    var devExamples = [Example]()
    let trainLines = try String(contentsOf: trainDataURL, encoding: .utf8).split { $0.isNewline }
    for line in trainLines.dropFirst() {
      let parts = line.split(separator: "\t")
        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
      let example = Example(
        id: "\(parts[1]):\(parts[2])",
        sentence1: parts[3],
        sentence2: parts[4],
        isParaphrase: parts[0] == "1")
      if devIds.contains(example.id) {
        devExamples.append(example)
      } else {
        trainExamples.append(example)
      }
    }
    self.trainExamples = trainExamples
    self.devExamples = devExamples

    // Load the test examples.
    let testLines = try String(contentsOf: testDataURL, encoding: .utf8).split { $0.isNewline }
    self.testExamples = testLines.dropFirst().enumerated().map { (index, line) in
      let parts = line.split { $0 == "\t" }
        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
      return Example(
        id: String(index),
        sentence1: parts[3],
        sentence2: parts[4],
        isParaphrase: parts[0] == "1")
    }

    self.maxSequenceLength = maxSequenceLength
    self.batchSize = batchSize

    // Create a function that converts examples to data batches.
    let exampleMapFn: (Example) -> DataBatch = { example -> DataBatch in
      let textBatch = architecture.preprocess(
        sequences: [example.sentence1, example.sentence2],
        maxSequenceLength: maxSequenceLength)
      return DataBatch(inputs: textBatch, labels: Tensor(example.isParaphrase ? 1 : 0))
    }

    // Create the data iterators used for training and evaluating.
    self.trainDataIterator = trainExamples.shuffled().makeIterator() // TODO: [RNG] Seed support.
      .repeated()
      .shuffled(bufferSize: 1000)
      .map(exampleMapFn)
      .grouped(
        keyFn: { $0.inputs.tokenIds.shape[1] / 10 },
        sizeFn: { key in batchSize / ((key + 1) * 10) },
        reduceFn: { DataBatch(
          inputs: padAndBatch(textBatches: $0.map { $0.inputs }),
          labels: Tensor.batch($0.map { $0.labels }))
        })
      .prefetched(count: 2)
    self.devDataIterator = devExamples.makeIterator()
      .map(exampleMapFn)
      .grouped(
        keyFn: { $0.inputs.tokenIds.shape[1] / 10 },
        sizeFn: { key in batchSize / ((key + 1) * 10) },
        reduceFn: { DataBatch(
          inputs: padAndBatch(textBatches: $0.map { $0.inputs }),
          labels: Tensor.batch($0.map { $0.labels }))
        })
      .prefetched(count: 2)
    self.testDataIterator = testExamples.makeIterator()
      .map(exampleMapFn)
      .grouped(
        keyFn: { $0.inputs.tokenIds.shape[1] / 10 },
        sizeFn: { key in batchSize / ((key + 1) * 10) },
        reduceFn: { DataBatch(
          inputs: padAndBatch(textBatches: $0.map { $0.inputs }),
          labels: Tensor.batch($0.map { $0.labels }))
        })
      .prefetched(count: 2)
  }
}

//===-----------------------------------------------------------------------------------------===//
// Data
//===-----------------------------------------------------------------------------------------===//

extension MRPC {
  /// MRPC example.
  public struct Example {
    public let id: String
    public let sentence1: String
    public let sentence2: String
    public let isParaphrase: Bool

    public init(id: String, sentence1: String, sentence2: String, isParaphrase: Bool) {
      self.id = id
      self.sentence1 = sentence1
      self.sentence2 = sentence2
      self.isParaphrase = isParaphrase
    }
  }

  /// MRPC data batch.
  public struct DataBatch: KeyPathIterable {
    public var inputs: TextBatch     // TODO: !!! Mutable in order to allow for batching.
    public var labels: Tensor<Int32> // TODO: !!! Mutable in order to allow for batching.

    public init(inputs: TextBatch, labels: Tensor<Int32>) {
      self.inputs = inputs
      self.labels = labels
    }
  }

  /// URL pointing to the downloadable text file that contains the train sentences.
  private static let trainDataURL: URL = URL(string: String(
    "https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_train.txt"))!

  /// URL pointing to the downloadable text file that contains the test sentences.
  private static let testDataURL: URL = URL(string: String(
    "https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_test.txt"))!

  /// URL pointing to the downloadable text file that contains the dev-set IDs.
  private static let devIdsURL: URL = URL(string: String(
    "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/" +
      "o/data%2Fmrpc_dev_ids.tsv?alt=media&token=ec5c0836-31d5-48f4-b431-7480817f1adc"))!
}
