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

public struct CoLA: Task {
  public let directoryURL: URL
  public let trainExamples: [Example]
  public let devExamples: [Example]
  public let testExamples: [Example]
  public let textTokenizer: FullTextTokenizer
  public let maxSequenceLength: Int
  public let batchSize: Int

  public let problem: Classification = Classification(
    context: .grammaticalCorrectness,
    concepts: [.positive, .negative])

  private typealias ExampleIterator = IndexingIterator<Array<Example>>
  private typealias RepeatExampleIterator = ShuffleIterator<RepeatIterator<ExampleIterator>>
  private typealias TrainDataIterator = MapIterator<RepeatExampleIterator, DataBatch>
  private typealias DevDataIterator = MapIterator<ExampleIterator, DataBatch>
  private typealias TestDataIterator = DevDataIterator

  private var trainDataIterator: TrainDataIterator
  private var devDataIterator: DevDataIterator
  private var testDataIterator: TestDataIterator

  public mutating func update<A: Architecture, O: Optimizer>(
    architecture: inout A,
    using optimizer: inout O
  ) -> Float where O.Model == A {
    let batch = trainDataIterator.next()!
    let textBatch = TextBatch(
      tokenIds: batch.tokenIds,
      tokenTypeIds: batch.tokenTypeIds,
      mask: batch.mask)
    let labelsBatch = batch.labelIds!
    let input = ArchitectureInput(text: textBatch)
    let problem = self.problem
    let (loss, gradient) = architecture.valueWithGradient {
      softmaxCrossEntropy(
        logits: $0.classify(input, problem: problem),
        labels: labelsBatch,
        reduction: { $0.mean() })
    }
    optimizer.update(&architecture, along: gradient)
    return loss.scalarized()
  }

  public func evaluate<A: Architecture>(using architecture: A) -> EvaluationResult {
    var devDataIterator = self.devDataIterator
    var devPredictedLabels = [Bool]()
    while let batch = devDataIterator.next() {
      let textBatch = TextBatch(
        tokenIds: batch.tokenIds,
        tokenTypeIds: batch.tokenTypeIds,
        mask: batch.mask)
      let input = ArchitectureInput(text: textBatch)
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

extension CoLA {
  public init(
    taskDirectoryURL: URL,
    textTokenizer: FullTextTokenizer,
    maxSequenceLength: Int,
    batchSize: Int
  ) throws {
    self.directoryURL = taskDirectoryURL.appendingPathComponent("CoLA")

    let dataURL = directoryURL.appendingPathComponent("data")
    let compressedDataURL = dataURL.appendingPathComponent("downloaded-data.zip")

    // Download the data, if necessary.
    try maybeDownload(from: CoLA.url, to: compressedDataURL)

    // Extract the data, if necessary.
    let extractedDirectoryURL = compressedDataURL.deletingPathExtension()
    if !FileManager.default.fileExists(atPath: extractedDirectoryURL.path) {
      try FileManager.default.unzipItem(at: compressedDataURL, to: extractedDirectoryURL)
    }

    // Load the data files into arrays of examples.
    let dataFilesURL = extractedDirectoryURL.appendingPathComponent("CoLA")
    self.trainExamples = try CoLA.load(
      fromFile: dataFilesURL.appendingPathComponent("train.tsv"),
      fileType: .train)
    self.devExamples = try CoLA.load(
      fromFile: dataFilesURL.appendingPathComponent("dev.tsv"),
      fileType: .dev)
    self.testExamples = try CoLA.load(
      fromFile: dataFilesURL.appendingPathComponent("test.tsv"),
      fileType: .test)

    self.textTokenizer = textTokenizer
    self.maxSequenceLength = maxSequenceLength
    self.batchSize = batchSize

    // Create a function that converts examples to data batches.
    let exampleMapFn = { example in
      CoLA.convertExampleToDataBatch(
        example,
        maxSequenceLength: maxSequenceLength,
        textTokenizer: textTokenizer)
    }

    // Create the data iterators used for training and evaluating.
    self.trainDataIterator = trainExamples.shuffled().makeIterator() // TODO: [RNG] Seed support.
      .repeated()
      .shuffled(bufferSize: 1000)
      .map(exampleMapFn)
    self.devDataIterator = devExamples.makeIterator().map(exampleMapFn)
    self.testDataIterator = testExamples.makeIterator().map(exampleMapFn)
  }

  /// Converts an example to a data batch.
  ///
  /// - Parameters:
  ///   - example: Example to convert.
  ///   - maxSequenceLength: Maximum allowed sequence length.
  ///   - textTokenizer: Text tokenizer to use for the conversion.
  ///
  /// - Returns: Data batch that corresponds to the provided example.
  private static func convertExampleToDataBatch(
    _ example: Example,
    maxSequenceLength: Int,
    textTokenizer: FullTextTokenizer
  ) -> DataBatch {
    let tokenized = preprocessText(
      sequences: [example.sentence],
      maxSequenceLength: maxSequenceLength,
      usingTokenizer: textTokenizer)
    return DataBatch(
      tokenIds: Tensor(tokenized.tokenIds.map(Int32.init)).expandingShape(at: 0),
      tokenTypeIds: Tensor(tokenized.tokenTypeIds.map(Int32.init)).expandingShape(at: 0),
      mask: Tensor(tokenized.mask.map { $0 ? 1 : 0 }).expandingShape(at: 0),
      labelIds: example.isAcceptable.map { Tensor($0 ? 1 : 0).expandingShape(at: 0) })
  }
}

//===-----------------------------------------------------------------------------------------===//
// Data
//===-----------------------------------------------------------------------------------------===//

extension CoLA {
  /// CoLA example.
  public struct Example {
    public let id: String
    public let sentence: String
    public let isAcceptable: Bool?

    public init(id: String, sentence: String, isAcceptable: Bool?) {
      self.id = id
      self.sentence = sentence
      self.isAcceptable = isAcceptable
    }
  }

  /// CoLA data batch.
  public struct DataBatch {
    public let tokenIds: Tensor<Int32>
    public let tokenTypeIds: Tensor<Int32>
    public let mask: Tensor<Int32>
    public let labelIds: Tensor<Int32>?

    public init(
      tokenIds: Tensor<Int32>,
      tokenTypeIds: Tensor<Int32>,
      mask: Tensor<Int32>,
      labelIds: Tensor<Int32>? = nil
    ) {
      self.tokenIds = tokenIds
      self.tokenTypeIds = tokenTypeIds
      self.mask = mask
      self.labelIds = labelIds
    }
  }
}

//===-----------------------------------------------------------------------------------------===//
// Evaluation
//===-----------------------------------------------------------------------------------------===//

extension CoLA {
  public struct EvaluationResult: Result {
    public let matthewsCorrelationCoefficient: Float
    public let accuracy: Float

    public init(matthewsCorrelationCoefficient: Float, accuracy: Float) {
      self.matthewsCorrelationCoefficient = matthewsCorrelationCoefficient
      self.accuracy = accuracy
    }

    public var summary: String {
      """
      Matthew's Correlation Coefficient: \(matthewsCorrelationCoefficient)
      Accuracy:                          \(accuracy)
      """
    }
  }

  public func evaluate(examples: [Example], predictions: [String: Bool]) -> EvaluationResult {
    let predictions = examples.map { predictions[$0.id]! }
    let groundTruth = examples.map { $0.isAcceptable! }
    return EvaluationResult(
      matthewsCorrelationCoefficient: NCA.matthewsCorrelationCoefficient(
        predictions: predictions,
        groundTruth: groundTruth),
      accuracy: NCA.accuracy(predictions: predictions, groundTruth: groundTruth))
  }
}

//===-----------------------------------------------------------------------------------------===//
// Data Loading
//===-----------------------------------------------------------------------------------------===//

extension CoLA {
  /// URL pointing to the downloadable ZIP file that contains the CoLA dataset.
  private static let url: URL = URL(string: String(
    "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/" +
      "o/data%2FCoLA.zip?alt=media&token=46d5e637-3411-4188-bc44-5809b5bfb5f4"))!

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
          id: "\(fileType.rawValue)-\(i)",
          sentence: lineParts[1],
          isAcceptable: nil)
      }
    }

    return lines.enumerated().map { (i, line) in
      let lineParts = line.components(separatedBy: "\t")
      return Example(
        id: "\(fileType.rawValue)-\(i)",
        sentence: lineParts[3],
        isAcceptable: lineParts[1] == "1")
    }
  }
}
