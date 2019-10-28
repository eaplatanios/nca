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

// TODO: !!! This is exactly the same as RTE and shares A LOT of code with CoLA and MRPC.

public struct WNLI: Task {
  public let directoryURL: URL
  public let trainExamples: [Example]
  public let devExamples: [Example]
  public let testExamples: [Example]
  public let textTokenizer: FullTextTokenizer
  public let maxSequenceLength: Int
  public let batchSize: Int

  public let problem: Classification = Classification(
    context: .entailment,
    concepts: [.neutral, .positive])

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
    let batch = withDevice(.cpu) { trainDataIterator.next()! }
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
    while let batch = withDevice(.cpu, perform: { devDataIterator.next() }) {
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

extension WNLI {
  public init(
    taskDirectoryURL: URL,
    textTokenizer: FullTextTokenizer,
    maxSequenceLength: Int,
    batchSize: Int
  ) throws {
    self.directoryURL = taskDirectoryURL.appendingPathComponent("WNLI")

    let dataURL = directoryURL.appendingPathComponent("data")
    let compressedDataURL = dataURL.appendingPathComponent("downloaded-data.zip")

    // Download the data, if necessary.
    try maybeDownload(from: WNLI.url, to: compressedDataURL)

    // Extract the data, if necessary.
    let extractedDirectoryURL = compressedDataURL.deletingPathExtension()
    if !FileManager.default.fileExists(atPath: extractedDirectoryURL.path) {
      try extract(zipFileAt: compressedDataURL, to: extractedDirectoryURL)
    }

    // Load the data files into arrays of examples.
    let dataFilesURL = extractedDirectoryURL.appendingPathComponent("WNLI")
    self.trainExamples = try WNLI.load(
      fromFile: dataFilesURL.appendingPathComponent("train.tsv"),
      fileType: .train)
    self.devExamples = try WNLI.load(
      fromFile: dataFilesURL.appendingPathComponent("dev.tsv"),
      fileType: .dev)
    self.testExamples = try WNLI.load(
      fromFile: dataFilesURL.appendingPathComponent("test.tsv"),
      fileType: .test)

    self.textTokenizer = textTokenizer
    self.maxSequenceLength = maxSequenceLength
    self.batchSize = batchSize

    // Create a function that converts examples to data batches.
    let exampleMapFn = { example in
      WNLI.convertExampleToBatch(
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
      sequences: [example.sentence1, example.sentence2],
      maxSequenceLength: maxSequenceLength,
      usingTokenizer: textTokenizer)
    return DataBatch(
      inputs: TextBatch(
        tokenIds: Tensor(tokenized.tokenIds.map(Int32.init)),
        tokenTypeIds: Tensor(tokenized.tokenTypeIds.map(Int32.init)),
        mask: Tensor(tokenized.mask.map { $0 ? 1 : 0 })),
      labels: example.entailment.map { Tensor($0 ? 1 : 0) })
  }
}

//===-----------------------------------------------------------------------------------------===//
// Data
//===-----------------------------------------------------------------------------------------===//

extension WNLI {
  /// WNLI example.
  public struct Example {
    public let id: String
    public let sentence1: String
    public let sentence2: String
    public let entailment: Bool?

    public init(id: String, sentence1: String, sentence2: String, entailment: Bool?) {
      self.id = id
      self.sentence1 = sentence1
      self.sentence2 = sentence2
      self.entailment = entailment
    }
  }

  /// WNLI data batch.
  public struct DataBatch: KeyPathIterable {
    public var inputs: TextBatch      // TODO: !!! Mutable in order to allow for batching.
    public var labels: Tensor<Int32>? // TODO: !!! Mutable in order to allow for batching.

    public init(inputs: TextBatch, labels: Tensor<Int32>?) {
      self.inputs = inputs
      self.labels = labels
    }
  }

  /// URL pointing to the downloadable ZIP file that contains the WNLI dataset.
  private static let url: URL = URL(string: String(
    "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/" +
      "o/data%2FWNLI.zip?alt=media&token=068ad0a0-ded7-4bd7-99a5-5e00222e0faf"))!

  internal enum FileType: String {
    case train = "train", dev = "dev", test = "test"
  }

  internal static func load(fromFile fileURL: URL, fileType: FileType) throws -> [Example] {
    let lines = try String(contentsOf: fileURL, encoding: .utf8).split { $0.isNewline }

    if fileType == .test {
      // The test data file has a header.
      return lines.dropFirst().enumerated().map { (i, line) in
        let lineParts = line.components(separatedBy: "\t")
        return Example(
          id: lineParts[0],
          sentence1: lineParts[1],
          sentence2: lineParts[2],
          entailment: nil)
      }
    }

    return lines.dropFirst().enumerated().map { (i, line) in
      let lineParts = line.components(separatedBy: "\t")
      return Example(
        id: lineParts[0],
        sentence1: lineParts[1],
        sentence2: lineParts[2],
        entailment: lineParts[3] == "1")
    }
  }
}

//===-----------------------------------------------------------------------------------------===//
// Evaluation
//===-----------------------------------------------------------------------------------------===//

extension WNLI {
  public struct EvaluationResult: Result {
    public let accuracy: Float

    public init(accuracy: Float) {
      self.accuracy = accuracy
    }

    public var summary: String {
      "Accuracy: \(accuracy)"
    }
  }

  public func evaluate(examples: [Example], predictions: [String: Bool]) -> EvaluationResult {
    let predictions = examples.map { predictions[$0.id]! }
    let groundTruth = examples.map { $0.entailment! }
    return EvaluationResult(
      accuracy: NCA.accuracy(predictions: predictions, groundTruth: groundTruth))
  }
}
