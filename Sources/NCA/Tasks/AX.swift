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
//import Foundation
//import TensorFlow
//
//public struct AX: Task {
//  public let directoryURL: URL
//  public let testExamples: [Example]
//  public let textTokenizer: FullTextTokenizer
//  public let maxSequenceLength: Int
//  public let batchSize: Int
//
//  public let problem: Classification = Classification(
//    context: .equivalence,
//    concepts: [.negative, .positive])
//
//  private typealias ExampleIterator = IndexingIterator<Array<Example>>
//  private typealias TestDataIterator = GroupedIterator<MapIterator<ExampleIterator, TextBatch>>
//
//  private var testDataIterator: TestDataIterator
//
//  public mutating func update<A: Architecture, O: Optimizer>(
//    architecture: inout A,
//    using optimizer: inout O
//  ) -> Float where O.Model == A { }
//
//  public func evaluate<A: Architecture>(using architecture: A) -> EvaluationResult {
//    var devDataIterator = self.devDataIterator
//    var devPredictedLabels = [Bool]()
//    while let batch = withDevice(.cpu, perform: { devDataIterator.next() }) {
//      let input = ArchitectureInput(text: batch)
//      let predictions = architecture.classify(input, problem: problem)
//      let predictedLabels = predictions.argmax(squeezingAxis: -1) .== 1
//      devPredictedLabels.append(contentsOf: predictedLabels.scalars)
//    }
//    return evaluate(
//      examples: devExamples,
//      predictions: [String: Bool](
//        uniqueKeysWithValues: zip(devExamples.map { $0.id }, devPredictedLabels)))
//  }
//}
//
//extension AX {
//  public init(
//    taskDirectoryURL: URL,
//    textTokenizer: FullTextTokenizer,
//    maxSequenceLength: Int,
//    batchSize: Int
//  ) throws {
//    self.directoryURL = taskDirectoryURL.appendingPathComponent("AX")
//
//    let dataURL = directoryURL.appendingPathComponent("data")
//    let compressedDataURL = dataURL.appendingPathComponent("downloaded-data.zip")
//
//    // Download the data, if necessary.
//    try maybeDownload(from: AX.url, to: compressedDataURL)
//
//    // Load the data files into arrays of examples.
//    let dataFilesURL = extractedDirectoryURL.appendingPathComponent("AX")
//    self.testExamples = try AX.load(
//      fromFile: dataFilesURL.appendingPathComponent("test.tsv"),
//      fileType: .test)
//
//    self.textTokenizer = textTokenizer
//    self.maxSequenceLength = maxSequenceLength
//    self.batchSize = batchSize
//
//    // Create a function that converts examples to data batches.
//    let exampleMapFn = { example in
//      AX.convertExampleToBatch(
//        example,
//        maxSequenceLength: maxSequenceLength,
//        textTokenizer: textTokenizer)
//    }
//
//    // Create the test data iterator.
//    self.testDataIterator = testExamples.makeIterator()
//      .map(exampleMapFn)
//      .grouped(
//        keyFn: { $0.inputs.tokenIds.shape[1] / 10 },
//        sizeFn: { key in batchSize / ((key + 1) * 10) },
//        reduceFn: DataBatch.batch)
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
//  ) -> TextBatch {
//    let tokenized = preprocessText(
//      sequences: [example.sentence1, example.sentence2],
//      maxSequenceLength: maxSequenceLength,
//      usingTokenizer: textTokenizer)
//    return TextBatch(
//      tokenIds: Tensor(tokenized.tokenIds.map(Int32.init)),
//      tokenTypeIds: Tensor(tokenized.tokenTypeIds.map(Int32.init)),
//      mask: Tensor(tokenized.mask.map { $0 ? 1 : 0 }))
//  }
//}
//
////===-----------------------------------------------------------------------------------------===//
//// Data
////===-----------------------------------------------------------------------------------------===//
//
//extension AX {
//  /// AX example.
//  public struct Example {
//    public let id: String
//    public let sentence1: String
//    public let sentence2: String
//
//    public init(id: String, sentence1: String, sentence2: String) {
//      self.id = id
//      self.sentence1 = sentence1
//      self.sentence2 = sentence2
//    }
//  }
//
//  /// URL pointing to the downloadable ZIP file that contains the AX dataset.
//  private static let url: URL = URL(string: String(
//    "https://storage.googleapis.com/mtl-sentence-representations.appspot.com/" +
//    "tsvsWithoutLabels%2FAX.tsv?GoogleAccessId=firebase-adminsdk-0khhl" +
//    "@mtl-sentence-representations.iam.gserviceaccount.com&Expires=2498860800&" +
//    "Signature=DuQ2CSPt2Yfre0C%2BiISrVYrIFaZH1Lc7hBVZDD4ZyR7fZYOMNOUGpi8QxBmTN" +
//      "OrNPjR3z1cggo7WXFfrgECP6FBJSsURv8Ybrue8Ypt%2FTPxbuJ0Xc2FhDi%2BarnecCBFO" +
//      "77RSbfuz%2Bs95hRrYhTnByqu3U%2FYZPaj3tZt5QdfpH2IUROY8LiBXoXS46LE%2FgOQc%" +
//      "2FKN%2BA9SoscRDYsnxHfG0IjXGwHN%2Bf88q6hOmAxeNPx6moDulUF6XMUAaXCSFU%2BnR" +
//      "O2RDL9CapWxj%2BDl7syNyHhB7987hZ80B%2FwFkQ3MEs8auvt5XW1%2Bd4aCU7ytgM69r8" +
//      "JDCwibfhZxpaa4gd50QXQ%3D%3D"))!
//
//  internal static func load(fromFile fileURL: URL) throws -> [Example] {
//    let lines = try String(contentsOf: fileURL, encoding: .utf8).split { $0.isNewline }
//
//    if fileType == .test {
//      // The test data file has a header.
//      return lines.dropFirst().enumerated().map { (i, line) in
//        let lineParts = line.split(separator: "\t")
//        return Example(
//          id: "\(fileType.rawValue)-\(i)",
//          question1: String(lineParts[1]),
//          question2: String(lineParts[2]),
//          equivalent: nil)
//      }
//    }
//
//    return lines.dropFirst().enumerated().compactMap { (i, line) in
//      let lineParts = line.split(separator: "\t")
//      return Example(
//        id: String(lineParts[0]),
//        sentence1: String(lineParts[1]),
//        sentence2: String(lineParts[2]))
//    }
//  }
//}
//
////===-----------------------------------------------------------------------------------------===//
//// Evaluation
////===-----------------------------------------------------------------------------------------===//
//
//extension AX {
//  public struct EvaluationResult: Result {
//    public let f1Score: Float
//    public let accuracy: Float
//
//    public init(f1Score: Float, accuracy: Float) {
//      self.f1Score = f1Score
//      self.accuracy = accuracy
//    }
//
//    public var summary: String {
//      "Accuracy: \(accuracy), F1 Score: \(f1Score)"
//    }
//  }
//
//  public func evaluate(examples: [Example], predictions: [String: Bool]) -> EvaluationResult {
//    let predictions = examples.map { predictions[$0.id]! }
//    let groundTruth = examples.map { $0.equivalent! }
//    return EvaluationResult(
//      f1Score: NCA.f1Score(predictions: predictions, groundTruth: groundTruth),
//      accuracy: NCA.accuracy(predictions: predictions, groundTruth: groundTruth))
//  }
//}
