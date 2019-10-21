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

public struct Span {
  public let start: Int
  public let end: Int

  public var length: Int { end - start + 1 }

  public init(start: Int, end: Int) {
    self.start = start
    self.end = end
  }
}

public struct SQuAD {
  public let version: Version
  public let directoryURL: URL

  public let trainDataset: Dataset
  public let devDataset: Dataset

  public init(version: Version, taskDirectoryURL: URL) throws {
    self.version = version
    self.directoryURL = {
      switch version {
      case .v1_1: return taskDirectoryURL.appendingPathComponent("SQuAD-v1.1")
      case .v2: return taskDirectoryURL.appendingPathComponent("SQuAD-v2")
      }
    }()

    let dataURL = directoryURL.appendingPathComponent("data")
    let trainDataURL = dataURL.appendingPathComponent("train.json")
    let devDataURL = dataURL.appendingPathComponent("dev.json")

    // Download the data, if necessary.
    try maybeDownload(from: version.jsonTrainURL, to: trainDataURL)
    try maybeDownload(from: version.jsonDevURL, to: devDataURL)

    // Load the data.
    let trainData = try Foundation.Data(contentsOf: trainDataURL)
    let devData = try Foundation.Data(contentsOf: devDataURL)

    // Decode the data assuming JSON format.
    self.trainDataset = try JSONDecoder().decode(Dataset.self, from: trainData)
    self.devDataset = try JSONDecoder().decode(Dataset.self, from: devData)

    // Convert the datasets to arrays of examples.
    let trainExamples = SQuAD.convertToExamples(dataset: trainDataset, isTrain: true)
    let devExamples = SQuAD.convertToExamples(dataset: devDataset, isTrain: false)

    print(trainExamples.count)
    print(devExamples.count)

    exit(0)
  }
}

extension SQuAD {
  /// SQuAD dataset version.
  public enum Version: String, Codable {
    case v1_1 = "1.1", v2 = "v2.0"

    public var jsonTrainURL: URL {
      switch self {
      case .v1_1:
        return URL(string: "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json")!
      case .v2:
        return URL(string: "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json")!
      }
    }

    public var jsonDevURL: URL {
      switch self {
      case .v1_1:
        return URL(string: "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json")!
      case .v2:
        return URL(string: "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json")!
      }
    }
  }

  /// SQuAD example.
  public struct Example {
    public let questionId: String
    public let question: String
    public let paragraphTokens: [String]
    public let originalAnswer: String?
    public let answerSpan: Span?
    public let isImpossible: Bool

    public init(
      questionId: String,
      question: String,
      paragraphTokens: [String],
      originalAnswer: String?,
      answerSpan: Span?,
      isImpossible: Bool
    ) {
      self.questionId = questionId
      self.question = question
      self.paragraphTokens = paragraphTokens
      self.originalAnswer = originalAnswer
      self.answerSpan = answerSpan
      self.isImpossible = isImpossible
    }
  }
}

//===-----------------------------------------------------------------------------------------===//
// Evaluation
//===-----------------------------------------------------------------------------------------===//

extension SQuAD {
  public struct EvaluationResult {
    public let f1Score: Float
    public let exactMatchScore: Float

    public init(f1Score: Float, exactMatchScore: Float) {
      self.f1Score = f1Score
      self.exactMatchScore = exactMatchScore
    }
  }

  public func evaluate(dataset: Dataset, predictions: [String: String]) -> EvaluationResult {
    var f1Score: Float = 0.0
    var exactMatchScore: Float = 0.0
    var questionCount: Int = 0
    dataset.documents.forEach { document in
      document.paragraphs.forEach { paragraph in
        paragraph.questions.forEach { question in
          questionCount += 1
          guard let prediction = predictions[question.id] else {
            logger.warning("Unanswered question with ID \(question.id) will receive a score of 0.")
            return
          }
          let groundTruths = question.answers.map { $0.text }
          let result = evaluate(prediction: prediction, groundTruths: groundTruths)
          f1Score += result.f1Score
          exactMatchScore += result.exactMatchScore
        }
      }
    }
    return EvaluationResult(
      f1Score: f1Score / Float(questionCount),
      exactMatchScore: exactMatchScore / Float(questionCount))
  }

  internal func evaluate(prediction: String, groundTruths: [String]) -> EvaluationResult {
    EvaluationResult(
      f1Score: groundTruths.map { computeF1Score(prediction: prediction, groundTruth: $0) }.max()!,
      exactMatchScore: groundTruths.map {
        prediction.normalized == $0.normalized ? Float(100.0) : Float(0.0)
      }.max()!)
  }

  internal func computeF1Score(prediction: String, groundTruth: String) -> Float {
    // Tokenize the predictions and the ground truth.
    let predictionTokens = prediction.normalized.components(separatedBy: .whitespacesAndNewlines)
    let groundTruthTokens = groundTruth.normalized.components(separatedBy: .whitespacesAndNewlines)

    // If either answer is empty, then F-1 is 1 if both are empty and 0 otherwise.
    if predictionTokens.isEmpty || groundTruthTokens.isEmpty {
      return Float(predictionTokens.isEmpty && groundTruthTokens.isEmpty ? 1 : 0)
    }

    // Compute the frequencies of the common tokens.
    let predictionCounts = Dictionary(predictionTokens.map { ($0, 1) }, uniquingKeysWith: +)
    let groundTruthCounts = Dictionary(groundTruthTokens.map { ($0, 1) }, uniquingKeysWith: +)
    let commonTokens = Set(predictionCounts.keys).intersection(Set(groundTruthCounts.keys))
    let commonCount = Float(commonTokens.map {
      predictionCounts[$0]! + groundTruthCounts[$0]!
    }.reduce(0, +))

    // If there are no common tokens, then the F-1 score is 0.
    if commonCount == 0.0 {
      return 0.0
    }

    let precision = commonCount / Float(predictionTokens.count)
    let recall = commonCount / Float(groundTruthTokens.count)
    return 100 * 2 * precision * recall / (precision + recall)
  }
}

extension String {
  /// Normalized form of the answer string for performing SQuAD evaluation.
  internal var normalized: String {
    lowercased()                    // Lower-case.
      .filter { !$0.isPunctuation } // Remove punctuation.
      .replacingOccurrences(        // Remove articles.
        of: #"\b(a|an|the)\b"#,
        with: " ",
        options: .regularExpression)
  }
}

//===-----------------------------------------------------------------------------------------===//
// Data Loading
//===-----------------------------------------------------------------------------------------===//

extension SQuAD {
  /// Converts a SQuAD dataset into an array of SQuAD examples.
  ///
  /// - Parameters:
  ///   - dataset: Dataset to convert.
  ///   - isTrain: Indicates whether this is a train dataset.
  internal static func convertToExamples(dataset: Dataset, isTrain: Bool) -> [Example] {
    dataset.documents.flatMap { document -> [Example] in
      document.paragraphs.flatMap { paragraph -> [Example] in
        // Tokenize the paragraph text and create a mapping from character to word offsets.
        let paragraphText = paragraph.context
        var paragraphTokens = [String]()
        var previousIsWhitespace = true
        let characterToWordOffset = paragraphText.unicodeScalars.map { character -> Int in
          if character.properties.isWhitespace || character == "\u{202f}" {
            previousIsWhitespace = true
          } else {
            if previousIsWhitespace {
              paragraphTokens.append(String(character))
            } else {
              paragraphTokens[paragraphTokens.count - 1] += String(character)
            }
            previousIsWhitespace = false
          }
          return paragraphTokens.count - 1
        }

        // Create an example for each question about this paragraph.
        return paragraph.questions.compactMap { question -> Example? in
          let isImpossible = isTrain && dataset.version == .v2 && question.isImpossible ?? false
          var originalAnswer: String? = nil
          var answerSpan: Span? = nil
          var skipQuestion = false

          if isTrain && !isImpossible {
            assert(
              question.answers.count == 1,
              "For training, each question should have exactly 1 answer.")
            let answer = question.answers.first!
            let answerOffset = answer.start
            let answerLength = answer.text.count
            let startPosition = characterToWordOffset[answerOffset]
            let endPosition = characterToWordOffset[answerOffset + answerLength - 1]
            originalAnswer = answer.text
            answerSpan = Span(start: startPosition, end: endPosition)

            // We only add answers where the text can be exactly recovered from the document. If
            // this cannot happen it is likely due to weird Unicode stuff and so we just skip the
            // example. Note that this means that while training, not every example is guaranteed
            // to be preserved.
            let actualText = paragraphTokens[startPosition...endPosition].joined(separator: " ")
            let cleanAnswerText = originalAnswer!.split { $0.isWhitespace }.joined(separator: " ")
            if !actualText.contains(cleanAnswerText) {
              skipQuestion = true
            }
          }

          return skipQuestion ? nil : Example(
            questionId: question.id,
            question: question.question,
            paragraphTokens: paragraphTokens,
            originalAnswer: originalAnswer,
            answerSpan: answerSpan,
            isImpossible: isImpossible)
        }
      }
    }
  }

  public struct Dataset: Codable {
    public let version: Version
    public let documents: [Document]

    enum CodingKeys: String, CodingKey {
      case version = "version"
      case documents = "data"
    }
  }

  public struct Document: Codable {
    public let title: String
    public let paragraphs: [Paragraph]

    enum CodingKeys: String, CodingKey {
      case title = "title"
      case paragraphs = "paragraphs"
    }
  }

  public struct Paragraph: Codable {
    public let questions: [Question]
    public let context: String

    enum CodingKeys: String, CodingKey {
      case questions = "qas"
      case context = "context"
    }
  }

  public struct Question: Codable {
    public let id: String
    public let question: String
    public let answers: [Answer]
    public let plausibleAnswers: [Answer]?
    public let isImpossible: Bool?

    enum CodingKeys: String, CodingKey {
      case id = "id"
      case question = "question"
      case answers = "answers"
      case plausibleAnswers = "plausible_answers"
      case isImpossible = "is_impossible"
    }
  }

  public struct Answer: Codable {
    public let text: String
    public let start: Int

    enum CodingKeys: String, CodingKey {
      case text = "text"
      case start = "answer_start"
    }
  }
}
