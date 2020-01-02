//// Copyright 2019, Emmanouil Antonios Platanios. All Rights Reserved.
////
//// Licensed under the Apache License, Version 2.0 (the "License"); you may not
//// use this file except in compliance with the License. You may obtain a copy of
//// the License at
////
////     http://www.apache.org/licenses/LICENSE-2.0
////
//// Unless required by applicable law or agreed to in writing, software
//// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
//// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
//// License for the specific language governing permissions and limitations under
//// the License.
//
//import Core
//import Foundation
//import TensorFlow
//
//public struct RACE { //: Task {
//  public let directoryURL: URL
//  public let trainExamples: [Example]
//  public let devExamples: [Example]
//  public let testExamples: [Example]
//  public let textTokenizer: FullTextTokenizer
//  public let maxSequenceLength: Int
//  public let batchSize: Int
//
////  private typealias ExampleIterator = IndexingIterator<Array<Example>>
////  private typealias RepeatExampleIterator = ShuffleIterator<RepeatIterator<ExampleIterator>>
////  private typealias TrainDataIterator = PrefetchIterator<GroupedIterator<MapIterator<RepeatExampleIterator, DataBatch>>>
////  private typealias DevDataIterator = PrefetchIterator<GroupedIterator<MapIterator<ExampleIterator, DataBatch>>>
////  private typealias TestDataIterator = DevDataIterator
////
////  private var trainDataIterator: TrainDataIterator
////  private var devDataIterator: DevDataIterator
////  private var testDataIterator: TestDataIterator
//
////  public mutating func update<A: Architecture, O: Core.Optimizer>(
////    architecture: inout A,
////    using optimizer: inout O
////  ) -> Float where O.Model == A {
////    let batch = withDevice(.cpu) { trainDataIterator.next()! }
////    let input = ArchitectureInput(text: batch.inputs)
////    let labels = Tensor<Float>(batch.labels!)
////    let (loss, gradient) = architecture.valueWithGradient {
////      sigmoidCrossEntropy(
////        logits: $0.score(input, context: .inputScoring, concept: .grammaticalCorrectness),
////        labels: labels,
////        reduction: { $0.mean() })
////    }
////    optimizer.update(&architecture, along: gradient)
////    return loss.scalarized()
////  }
////
////  public func evaluate<A: Architecture>(using architecture: A) -> [String: Float] {
////    var devDataIterator = self.devDataIterator.copy()
////    var devPredictedLabels = [Bool]()
////    var devGroundTruth = [Bool]()
////    while let batch = withDevice(.cpu, perform: { devDataIterator.next() }) {
////      let input = ArchitectureInput(text: batch.inputs)
////      let predictions = architecture.score(
////        input,
////        context: .inputScoring,
////        concept: .grammaticalCorrectness)
////      let predictedLabels = sigmoid(predictions) .>= 0.5
////      devPredictedLabels.append(contentsOf: predictedLabels.scalars)
////      devGroundTruth.append(contentsOf: batch.labels!.scalars.map { $0 == 1 })
////    }
////    return [
////      "matthewsCorrelationCoefficient": NCA.matthewsCorrelationCoefficient(
////        predictions: devPredictedLabels,
////        groundTruth: devGroundTruth)]
////  }
//}
//
//extension RACE {
//  public enum Dataset {
//    case all
//    case middle
//    case high
//
//    /// Returns all the data files stored in the provided directory, for this dataset type.
//    internal func files(in directory: URL) throws -> [URL] {
//      var directories = [URL]()
//      switch self {
//      case .all:
//        directories.append(directory.appendingPathComponent("middle"))
//        directories.append(directory.appendingPathComponent("high"))
//      case .middle: directories.append(directory.appendingPathComponent("middle"))
//      case .high: directories.append(directory.appendingPathComponent("high"))
//      }
//      return try directories.flatMap { directory -> [URL] in
//        let directoryContents = try FileManager.default.contentsOfDirectory(
//          at: directory, includingPropertiesForKeys: nil)
//        return directoryContents.filter { $0.pathExtension == "txt" }
//      }
//    }
//  }
//
//  public init(
//    taskDirectoryURL: URL,
//    textTokenizer: FullTextTokenizer,
//    maxSequenceLength: Int,
//    batchSize: Int,
//    dataset: Dataset = .all
//  ) throws {
//    self.directoryURL = taskDirectoryURL.appendingPathComponent("RACE")
//
//    let dataURL = directoryURL.appendingPathComponent("data")
//    let compressedDataURL = dataURL.appendingPathComponent("downloaded-data.tar.gz")
//
//    // Download the data, if necessary.
//    try maybeDownload(from: RACE.url, to: compressedDataURL)
//
//    // Extract the data, if necessary.
//    let extractedDirectoryURL = dataURL.appendingPathComponent("downloaded-data")
//    if !FileManager.default.fileExists(atPath: extractedDirectoryURL.path) {
//      try extract(tarGZippedFileAt: compressedDataURL, to: extractedDirectoryURL)
//    }
//
//    // Load the data files into arrays of examples.
//    let dataFilesURL = extractedDirectoryURL.appendingPathComponent("RACE")
//    self.trainExamples = try RACE.load(
//      fromDirectory: dataFilesURL.appendingPathComponent("train"),
//      fileType: .train,
//      dataset: dataset)
//    self.devExamples = try RACE.load(
//      fromDirectory: dataFilesURL.appendingPathComponent("dev"),
//      fileType: .dev,
//      dataset: dataset)
//    self.testExamples = try RACE.load(
//      fromDirectory: dataFilesURL.appendingPathComponent("test"),
//      fileType: .test,
//      dataset: dataset)
//
//    let counts = devExamples.map { textTokenizer.tokenize($0.article).count }
//    dump(counts)
//    print(Float(counts.reduce(0, +)) / Float(counts.count))
//    print(counts.count)
//    print(counts.filter { $0 < 512 }.count)
//    print(counts.filter { $0 < 384 }.count)
//
//    exit(0)
//
//    self.textTokenizer = textTokenizer
//    self.maxSequenceLength = maxSequenceLength
//    self.batchSize = batchSize
//
////    // Create a function that converts examples to data batches.
////    let exampleMapFn = { example in
////      RACE.convertExampleToBatch(
////        example,
////        maxSequenceLength: maxSequenceLength,
////        textTokenizer: textTokenizer)
////    }
////
////    // Create the data iterators used for training and evaluating.
////    self.trainDataIterator = trainExamples.shuffled().makeIterator() // TODO: [RNG] Seed support.
////      .repeated()
////      .shuffled(bufferSize: 1000)
////      .map(exampleMapFn)
////      .grouped(
////        keyFn: { $0.inputs.tokenIds.shape[1] / 10 },
////        sizeFn: { key in batchSize / ((key + 1) * 10) },
////        reduceFn: { DataBatch(
////          inputs: padAndBatch(textBatches: $0.map { $0.inputs }),
////          labels: Tensor.batch($0.map { $0.labels! }))
////        })
////      .prefetched(count: 10)
////    self.devDataIterator = devExamples.makeIterator()
////      .map(exampleMapFn)
////      .grouped(
////        keyFn: { $0.inputs.tokenIds.shape[1] / 10 },
////        sizeFn: { key in batchSize / ((key + 1) * 10) },
////        reduceFn: { DataBatch(
////          inputs: padAndBatch(textBatches: $0.map { $0.inputs }),
////          labels: Tensor.batch($0.map { $0.labels! }))
////        })
////      .prefetched(count: 10)
////    self.testDataIterator = testExamples.makeIterator()
////      .map(exampleMapFn)
////      .grouped(
////        keyFn: { $0.inputs.tokenIds.shape[1] / 10 },
////        sizeFn: { key in batchSize / ((key + 1) * 10) },
////        reduceFn: { DataBatch(
////          inputs: padAndBatch(textBatches: $0.map { $0.inputs }),
////          labels: nil)
////        })
////      .prefetched(count: 10)
//  }
//
//  /// Converts an example to a data batch.
//  ///
//  /// - Parameters:
//  ///   - example: Example to convert.
//  ///   - maxSequenceLength: Maximum allowed sequence length.
//  ///   - textTokenizer: Text tokenizer to use for the conversion.
//  ///
//  /// - Returns: Data batch that corresponds to the provided example.
//  private static func convertExampleToBatch(
//    _ example: Example,
//    maxSequenceLength: Int,
//    textTokenizer: FullTextTokenizer
//  ) -> DataBatch {
//    let tokenized = preprocessText(
//      sequences: [example.article],
//      maxSequenceLength: maxSequenceLength,
//      usingTokenizer: textTokenizer)
//    return DataBatch(
//      inputs: TextBatch(
//        tokenIds: Tensor(tokenized.tokenIds.map(Int32.init)),
//        tokenTypeIds: Tensor(tokenized.tokenTypeIds.map(Int32.init)),
//        mask: Tensor(tokenized.mask.map { $0 ? 1 : 0 })),
//      labels: nil) // example.isAcceptable.map { Tensor($0 ? 1 : 0) })
//  }
//}
//
////===-----------------------------------------------------------------------------------------===//
//// Data
////===-----------------------------------------------------------------------------------------===//
//
//extension RACE {
//  /// RACE example.
//  public struct Example: Codable {
//    public let id: String
//    public let article: String
//    public let questions: [String]
//    public let options: [[String]]
//    public let answers: [String]?
//
//    public init(
//      id: String,
//      article: String,
//      questions: [String],
//      options: [[String]],
//      answers: [String]?
//    ) {
//      self.id = id
//      self.article = article
//      self.questions = questions
//      self.options = options
//      self.answers = answers
//    }
//  }
//
//  /// RACE data batch.
//  public struct DataBatch: KeyPathIterable {
//    public var inputs: TextBatch      // TODO: !!! Mutable in order to allow for batching.
//    public var labels: Tensor<Int32>? // TODO: !!! Mutable in order to allow for batching.
//
//    public init(inputs: TextBatch, labels: Tensor<Int32>?) {
//      self.inputs = inputs
//      self.labels = labels
//    }
//  }
//
//  /// URL pointing to the downloadable ZIP file that contains the RACE dataset.
//  private static let url: URL = URL(string: String(
//    "http://www.cs.cmu.edu/~glai1/data/race/RACE.tar.gz"))!
//
//  internal enum FileType: String {
//    case train = "train", dev = "dev", test = "test"
//  }
//
//  internal static func load(
//    fromDirectory directory: URL,
//    fileType: FileType,
//    dataset: Dataset
//  ) throws -> [Example] {
//    let files = try dataset.files(in: directory)
//    return try files.map {
//      try Example(fromJson: try String(contentsOfFile: $0.path, encoding: .utf8))
//    }
//  }
//}
//
//extension RACE.Example {
//  internal init(fromJson json: String) throws {
//    self = try JSONDecoder().decode(RACE.Example.self, from: json.data(using: .utf8)!)
//  }
//
//  internal func toJson(pretty: Bool = true) throws -> String {
//    let encoder = JSONEncoder()
//    if pretty {
//      encoder.outputFormatting = .prettyPrinted
//    }
//    let data = try encoder.encode(self)
//    return String(data: data, encoding: .utf8)!
//  }
//}
