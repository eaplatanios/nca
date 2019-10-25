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
import ZIPFoundation

public struct MRPC: Task {
  public let directoryURL: URL
  public let trainExamples: [Example]
  public let devExamples: [Example]
  public let testExamples: [Example]
  public let textTokenizer: FullTextTokenizer
  public let maxSequenceLength: Int
  public let batchSize: Int

  public let problem: Classification = Classification(
    context: .paraphrasing,
    concepts: [.negative, .positive])

  private typealias ExampleIterator = IndexingIterator<Array<Example>>
  private typealias RepeatExampleIterator = ShuffleIterator<RepeatIterator<ExampleIterator>>
  private typealias TrainDataIterator = BatchIterator<MapIterator<RepeatExampleIterator, DataBatch>>
  private typealias DevDataIterator = BatchIterator<MapIterator<ExampleIterator, DataBatch>>
  private typealias TestDataIterator = DevDataIterator

  private var trainDataIterator: TrainDataIterator
  private var devDataIterator: DevDataIterator
  private var testDataIterator: TestDataIterator

  public mutating func update<A: Architecture, O: Optimizer>(
    architecture: inout A,
    using optimizer: inout O
  ) -> Float where O.Model == A {
    let batch = trainDataIterator.next()!
    let input = ArchitectureInput(text: batch.inputs)
    let problem = self.problem
    let labels = batch.labels!
    let (loss, gradient) = architecture.valueWithGradient {
      softmaxCrossEntropy(
        logits: $0.classify(input, problem: problem),
        labels: labels,
        reduction: { $0.mean() })
    }
    optimizer.update(&architecture, along: gradient)
    return loss.scalarized()
  }

  public func evaluate<A: Architecture>(using architecture: A) -> EvaluationResult {
    var devDataIterator = self.devDataIterator
    var devPredictedLabels = [Bool]()
    while let batch = devDataIterator.next() {
      let input = ArchitectureInput(text: batch.inputs)
      let predictions = architecture.classify(input, problem: problem)
      let predictedLabels = predictions.argmax(squeezingAxis: -1) .== 1
      devPredictedLabels.append(contentsOf: predictedLabels.scalars)
    }
    return evaluate(
      examples: devExamples,
      predictions: [String: Bool](
        uniqueKeysWithValues: zip(devExamples.map { $0.id }, devPredictedLabels)))
  }
}

extension MRPC {
  public init(
    taskDirectoryURL: URL,
    textTokenizer: FullTextTokenizer,
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
        ids: (parts[1], parts[2]),
        sentences: (parts[3], parts[4]),
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
    self.testExamples = testLines.dropFirst().map { line in
      let parts = line.split { $0 == "\t" }
        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
      return Example(
        ids: (parts[1], parts[2]),
        sentences: (parts[3], parts[4]),
        isParaphrase: parts[0] == "1")
    }

    self.textTokenizer = textTokenizer
    self.maxSequenceLength = maxSequenceLength
    self.batchSize = batchSize

    // Create a function that converts examples to data batches.
    let exampleMapFn = { example in
      MRPC.convertExampleToBatch(
        example,
        maxSequenceLength: maxSequenceLength,
        textTokenizer: textTokenizer)
    }

    // Create the data iterators used for training and evaluating.
    self.trainDataIterator = trainExamples.shuffled().makeIterator() // TODO: [RNG] Seed support.
      .repeated()
      .shuffled(bufferSize: 1000)
      .map(exampleMapFn)
      .batched(batchSize: batchSize)
    self.devDataIterator = devExamples.makeIterator()
      .map(exampleMapFn)
      .batched(batchSize: batchSize)
    self.testDataIterator = testExamples.makeIterator()
      .map(exampleMapFn)
      .batched(batchSize: batchSize)
  }

  /// Converts an example to a data batch.
  ///
  /// - Parameters:
  ///   - example: Example to convert.
  ///   - maxSequenceLength: Maximum allowed sequence length.
  ///   - textTokenizer: Text tokenizer to use for the conversion.
  ///
  /// - Returns: Data batch that corresponds to the provided example.
  private static func convertExampleToBatch(
    _ example: Example,
    maxSequenceLength: Int,
    textTokenizer: FullTextTokenizer
  ) -> DataBatch {
    let tokenized = preprocessText(
      sequences: [example.sentences.0, example.sentences.1],
      maxSequenceLength: maxSequenceLength,
      usingTokenizer: textTokenizer)
    return DataBatch(
      inputs: TextBatch(
        tokenIds: Tensor(tokenized.tokenIds.map(Int32.init)),
        tokenTypeIds: Tensor(tokenized.tokenTypeIds.map(Int32.init)),
        mask: Tensor(tokenized.mask.map { $0 ? 1 : 0 })),
      labels: example.isParaphrase.map { Tensor($0 ? 1 : 0) })
  }
}

//===-----------------------------------------------------------------------------------------===//
// Data
//===-----------------------------------------------------------------------------------------===//

extension MRPC {
  /// MRPC example.
  public struct Example {
    public let ids: (String, String)
    public let sentences: (String, String)
    public let isParaphrase: Bool?

    public var id: String { "\(ids.0):\(ids.1)" }

    public init(ids: (String, String), sentences: (String, String), isParaphrase: Bool?) {
      self.ids = ids
      self.sentences = sentences
      self.isParaphrase = isParaphrase
    }
  }

  /// MRPC data batch.
  public struct DataBatch: KeyPathIterable {
    public var inputs: TextBatch      // TODO: !!! Mutable in order to allow for batching.
    public var labels: Tensor<Int32>? // TODO: !!! Mutable in order to allow for batching.

    public init(inputs: TextBatch, labels: Tensor<Int32>?) {
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

//===-----------------------------------------------------------------------------------------===//
// Evaluation
//===-----------------------------------------------------------------------------------------===//

extension MRPC {
  public struct EvaluationResult: Result {
    public let f1Score: Float
    public let accuracy: Float

    public init(f1Score: Float, accuracy: Float) {
      self.f1Score = f1Score
      self.accuracy = accuracy
    }

    public var summary: String {
      "Accuracy: \(accuracy), F1 Score: \(f1Score)"
    }
  }

  public func evaluate(examples: [Example], predictions: [String: Bool]) -> EvaluationResult {
    let predictions = examples.map { predictions[$0.id]! }
    let groundTruth = examples.map { $0.isParaphrase! }
    return EvaluationResult(
      f1Score: NCA.f1Score(predictions: predictions, groundTruth: groundTruth),
      accuracy: NCA.accuracy(predictions: predictions, groundTruth: groundTruth))
  }
}
