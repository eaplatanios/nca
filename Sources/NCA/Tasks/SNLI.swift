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

// TODO: !!! This is almost exactly the same as MNLI and shares A LOT of code with CoLA and MRPC.

public struct SNLI: Task {
  public let directoryURL: URL
  public let trainExamples: [Example]
  public let devExamples: [Example]
  public let testExamples: [Example]
  public let textTokenizer: FullTextTokenizer
  public let maxSequenceLength: Int
  public let batchSize: Int

  public let problem: Classification = Classification(
    context: .entailment,
    concepts: [.positive, .negative, .neutral])

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

  public func evaluate<A: Architecture>(using architecture: A) -> [String: Float] {
    var devDataIterator = self.devDataIterator.copy()
    var devPredictedLabels = [Int32]()
    var devGroundTruth = [Int32]()
    while let batch = withDevice(.cpu, perform: { devDataIterator.next() }) {
      let input = ArchitectureInput(text: batch.inputs)
      let predictions = architecture.classify(input, problem: problem)
      let predictedLabels = predictions.argmax(squeezingAxis: -1)
      devPredictedLabels.append(contentsOf: predictedLabels.scalars)
      devGroundTruth.append(contentsOf: batch.labels!.scalars)
    }
    return ["accuracy": NCA.accuracy(predictions: devPredictedLabels, groundTruth: devGroundTruth)]
  }
}

extension SNLI {
  public init(
    taskDirectoryURL: URL,
    textTokenizer: FullTextTokenizer,
    maxSequenceLength: Int,
    batchSize: Int
  ) throws {
    self.directoryURL = taskDirectoryURL.appendingPathComponent("SNLI")

    let dataURL = directoryURL.appendingPathComponent("data")
    let compressedDataURL = dataURL.appendingPathComponent("downloaded-data.zip")

    // Download the data, if necessary.
    try maybeDownload(from: SNLI.url, to: compressedDataURL)

    // Extract the data, if necessary.
    let extractedDirectoryURL = compressedDataURL.deletingPathExtension()
    if !FileManager.default.fileExists(atPath: extractedDirectoryURL.path) {
      try extract(zipFileAt: compressedDataURL, to: extractedDirectoryURL)
    }

    // Load the data files into arrays of examples.
    let dataFilesURL = extractedDirectoryURL.appendingPathComponent("SNLI")
    self.trainExamples = try SNLI.load(
      fromFile: dataFilesURL.appendingPathComponent("train.tsv"),
      fileType: .train)
    self.devExamples = try SNLI.load(
      fromFile: dataFilesURL.appendingPathComponent("dev.tsv"),
      fileType: .dev)
    self.testExamples = try SNLI.load(
      fromFile: dataFilesURL.appendingPathComponent("test.tsv"),
      fileType: .test)

    self.textTokenizer = textTokenizer
    self.maxSequenceLength = maxSequenceLength
    self.batchSize = batchSize

    // Create a function that converts examples to data batches.
    let exampleMapFn = { example in
      SNLI.convertExampleToBatch(
        example,
        maxSequenceLength: maxSequenceLength,
        textTokenizer: textTokenizer)
    }

    // Create the data iterators used for training and evaluating.
    self.trainDataIterator = trainExamples.shuffled().makeIterator() // TODO: [RNG] Seed support.
      .repeated()
      .shuffled(bufferSize: 1000)
      .map(exampleMapFn)
      .grouped(
        keyFn: { $0.inputs.tokenIds.shape[0] / 10 },
        sizeFn: { key in batchSize / ((key + 1) * 10) },
        reduceFn: { DataBatch(
          inputs: padAndBatch(textBatches: $0.map { $0.inputs }),
          labels: Tensor.batch($0.map { $0.labels! }))
        })
      .prefetched(count: 2)
    self.devDataIterator = devExamples.makeIterator()
      .map(exampleMapFn)
      .grouped(
        keyFn: { $0.inputs.tokenIds.shape[0] / 10 },
        sizeFn: { key in batchSize / ((key + 1) * 10) },
        reduceFn: { DataBatch(
          inputs: padAndBatch(textBatches: $0.map { $0.inputs }),
          labels: Tensor.batch($0.map { $0.labels! }))
        })
      .prefetched(count: 2)
    self.testDataIterator = testExamples.makeIterator()
      .map(exampleMapFn)
      .grouped(
        keyFn: { $0.inputs.tokenIds.shape[0] / 10 },
        sizeFn: { key in batchSize / ((key + 1) * 10) },
        reduceFn: { DataBatch(
          inputs: padAndBatch(textBatches: $0.map { $0.inputs }),
          labels: nil)
        })
      .prefetched(count: 2)
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
      labels: example.entailment.map { entailment in Tensor<Int32>({ () -> Int32 in
        switch entailment {
        case .entailment: return 0
        case .contradiction: return 1
        case .neutral: return 2
        }
      }())})
  }
}

//===-----------------------------------------------------------------------------------------===//
// Data
//===-----------------------------------------------------------------------------------------===//

extension SNLI {
  public enum Entailment: Int32 {
    case entailment = 0, contradiction = 1, neutral = 2
  }

  /// SNLI example.
  public struct Example {
    public let id: String
    public let sentence1: String
    public let sentence2: String
    public let entailment: Entailment?

    public init(id: String, sentence1: String, sentence2: String, entailment: Entailment?) {
      self.id = id
      self.sentence1 = sentence1
      self.sentence2 = sentence2
      self.entailment = entailment
    }
  }

  /// SNLI data batch.
  public struct DataBatch: KeyPathIterable {
    public var inputs: TextBatch      // TODO: !!! Mutable in order to allow for batching.
    public var labels: Tensor<Int32>? // TODO: !!! Mutable in order to allow for batching.

    public init(inputs: TextBatch, labels: Tensor<Int32>?) {
      self.inputs = inputs
      self.labels = labels
    }
  }

  /// URL pointing to the downloadable ZIP file that contains the SNLI dataset.
  private static let url: URL = URL(string: String(
    "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/" +
      "o/data%2FSNLI.zip?alt=media&token=4afcfbb2-ff0c-4b2d-a09a-dbf07926f4df"))!

  internal enum FileType: String {
    case train = "train", dev = "dev", test = "test"
  }

  internal static func load(fromFile fileURL: URL, fileType: FileType) throws -> [Example] {
    let lines = try parse(tsvFileAt: fileURL)

    if fileType == .test {
      // The test data file has a header.
      return lines.dropFirst().enumerated().map { (i, lineParts) in
        Example(
          id: lineParts[0],
          sentence1: lineParts[8],
          sentence2: lineParts[9],
          entailment: nil)
      }
    }

    return lines.dropFirst().enumerated().map { (i, lineParts) in
      Example(
        id: lineParts[0],
        sentence1: lineParts[8],
        sentence2: lineParts[9],
        entailment: lineParts.last! == "entailment" ?
          .entailment :
          lineParts.last! == "contradiction" ? .contradiction : .neutral)
    }
  }
}
