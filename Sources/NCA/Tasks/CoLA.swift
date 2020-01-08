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

import Core
import Foundation
import TensorFlow

public struct CoLA: Task {
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

  public let problem: Problem = .grammar
  public let concepts: [Concept] = [.grammaticallyIncorrect, .grammaticallyCorrect]

  public mutating func update<A: Architecture, O: Core.Optimizer>(
    architecture: inout A,
    using optimizer: inout O
  ) -> Float where O.Model == A {
    let batch = withDevice(.cpu) { trainDataIterator.next()! }
    let input = ArchitectureInput(text: batch.inputs)
    let labels = batch.labels!
    let problem = self.problem
    let concepts = self.concepts
    return withLearningPhase(.training) {
      let (loss, gradient) = valueWithGradient(at: architecture) {
        softmaxCrossEntropy(
          logits: $0.classify(input, problem: problem, concepts: concepts),
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
      let predictions = architecture.classify(input, problem: problem, concepts: concepts)
      let predictedLabels = predictions.argmax(squeezingAxis: -1) .== 1
      devPredictedLabels.append(contentsOf: predictedLabels.scalars)
      devGroundTruth.append(contentsOf: batch.labels!.scalars.map { $0 == 1 })
    }
    return [
      "matthewsCorrelationCoefficient": NCA.matthewsCorrelationCoefficient(
        predictions: devPredictedLabels,
        groundTruth: devGroundTruth)]
  }
}

extension CoLA {
  public init<A: Architecture>(
    for architecture: A,
    taskDirectoryURL: URL,
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
      try extract(zipFileAt: compressedDataURL, to: extractedDirectoryURL)
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

    self.maxSequenceLength = maxSequenceLength
    self.batchSize = batchSize

    // Create a function that converts examples to data batches.
    let exampleMapFn: (Example) -> DataBatch = { example -> DataBatch in
      let textBatch = architecture.preprocess(
        sequences: [example.sentence],
        maxSequenceLength: maxSequenceLength)
      return DataBatch(inputs: textBatch, labels: example.isAcceptable.map { Tensor($0 ? 1 : 0) })
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
          labels: Tensor.batch($0.map { $0.labels! }))
        })
      .prefetched(count: 2)
    self.devDataIterator = devExamples.makeIterator()
      .map(exampleMapFn)
      .grouped(
        keyFn: { $0.inputs.tokenIds.shape[1] / 10 },
        sizeFn: { key in batchSize / ((key + 1) * 10) },
        reduceFn: { DataBatch(
          inputs: padAndBatch(textBatches: $0.map { $0.inputs }),
          labels: Tensor.batch($0.map { $0.labels! }))
        })
      .prefetched(count: 2)
    self.testDataIterator = testExamples.makeIterator()
      .map(exampleMapFn)
      .grouped(
        keyFn: { $0.inputs.tokenIds.shape[1] / 10 },
        sizeFn: { key in batchSize / ((key + 1) * 10) },
        reduceFn: { DataBatch(
          inputs: padAndBatch(textBatches: $0.map { $0.inputs }),
          labels: nil)
        })
      .prefetched(count: 2)
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
  public struct DataBatch: KeyPathIterable {
    public var inputs: TextBatch      // TODO: !!! Mutable in order to allow for batching.
    public var labels: Tensor<Int32>? // TODO: !!! Mutable in order to allow for batching.

    public init(inputs: TextBatch, labels: Tensor<Int32>?) {
      self.inputs = inputs
      self.labels = labels
    }
  }

  /// URL pointing to the downloadable ZIP file that contains the CoLA dataset.
  private static let url: URL = URL(string: String(
    "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/" +
      "o/data%2FCoLA.zip?alt=media&token=46d5e637-3411-4188-bc44-5809b5bfb5f4"))!

  internal enum FileType: String {
    case train = "train", dev = "dev", test = "test"
  }

  internal static func load(fromFile fileURL: URL, fileType: FileType) throws -> [Example] {
    let lines = try parse(tsvFileAt: fileURL)

    if fileType == .test {
      // The test data file has a header.
      return lines.dropFirst().enumerated().map { (i, lineParts) in
        Example(id: lineParts[0], sentence: lineParts[1], isAcceptable: nil)
      }
    }

    return lines.enumerated().map { (i, lineParts) in
      Example(id: lineParts[0], sentence: lineParts[3], isAcceptable: lineParts[1] == "1")
    }
  }
}
