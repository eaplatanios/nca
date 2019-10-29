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

import Logging
import NCA
import Foundation
import TensorFlow

let logger = Logger(label: "NCA Experiment")

/// Helper string interpolation extensions for logging training progress.
extension String.StringInterpolation {
  mutating func appendInterpolation(step value: Int) {
    appendLiteral(String(format: "Step %d", value).leftPadding(toLength: 10, withPad: " "))
  }

  mutating func appendInterpolation(task value: String) {
    appendLiteral(value.leftPadding(toLength: 5, withPad: " "))
  }

  mutating func appendInterpolation(metricName value: String) {
    appendLiteral(value.leftPadding(toLength: 30, withPad: " "))
  }

  mutating func appendInterpolation(metricValue value: Float) {
    appendLiteral(String(format: "%3.2f", value * 100).leftPadding(toLength: 6, withPad: " "))
  }

  mutating func appendInterpolation(loss value: Float) {
    appendLiteral(String(format: "%.4f", value))
  }
}

extension String {
  func leftPadding(toLength: Int, withPad character: Character) -> String {
    if count < toLength {
      return String(repeatElement(character, count: toLength - count)) + self
    } else {
      return String(self[index(self.startIndex, offsetBy: count - toLength)...])
    }
  }
}

let currentDir = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
let ncaDir = currentDir.appendingPathComponent("temp")
let modulesDir = ncaDir.appendingPathComponent("modules")
let tasksDir = ncaDir.appendingPathComponent("tasks")

let bertDir = modulesDir
  .appendingPathComponent("text")
  .appendingPathComponent("bert")
let bertPreTrainedModel = BERT.PreTrainedModel.base(cased: false, multilingual: false)
try bertPreTrainedModel.maybeDownload(to: bertDir)
let bertConfigurationURL = bertDir
  .appendingPathComponent(bertPreTrainedModel.name)
  .appendingPathComponent("bert_config.json")
let vocabularyURL = bertDir
  .appendingPathComponent(bertPreTrainedModel.name)
  .appendingPathComponent("vocab.txt")
let vocabulary = try! Vocabulary(fromFile: vocabularyURL)
let bertConfiguration = try! BERT.Configuration(fromFile: bertConfigurationURL)

// let albertDir = modulesDir
//   .appendingPathComponent("text")
//   .appendingPathComponent("albert")
// let albertPreTrainedModel = ALBERT.PreTrainedModel.base
// try albertPreTrainedModel.maybeDownload(to: albertDir)
// let vocabularyURL = albertDir
//   .appendingPathComponent(albertPreTrainedModel.name)
//   .appendingPathComponent("assets")
//   .appendingPathComponent("30k-clean.model")
// let vocabulary = try! Vocabulary(fromSentencePieceModel: vocabularyURL)
// let albertConfiguration = albertPreTrainedModel.configuration

let textTokenizer = FullTextTokenizer(
  caseSensitive: false,
  vocabulary: vocabulary,
  unknownToken: "[UNK]",
  maxTokenLength: bertConfiguration.maxSequenceLength)

let maxSequenceLength = 128 // bertConfiguration.maxSequenceLength
let taskInitializers: [() -> (String, Task)] = [
  { () in
    ("MRPC", try! MRPC(
      taskDirectoryURL: tasksDir,
      textTokenizer: textTokenizer,
      maxSequenceLength: maxSequenceLength,
      batchSize: 1024))
  },
  { () in
    ("CoLA", try! CoLA(
      taskDirectoryURL: tasksDir,
      textTokenizer: textTokenizer,
      maxSequenceLength: maxSequenceLength,
      batchSize: 1024))
  },
  { () in
    ("RTE", try! RTE(
      taskDirectoryURL: tasksDir,
      textTokenizer: textTokenizer,
      maxSequenceLength: maxSequenceLength,
      batchSize: 1024))
  },
  { () in
    ("SST", try! SST(
      taskDirectoryURL: tasksDir,
      textTokenizer: textTokenizer,
      maxSequenceLength: maxSequenceLength,
      batchSize: 1024))
  },
  { () in
    ("STS", try! STS(
      taskDirectoryURL: tasksDir,
      textTokenizer: textTokenizer,
      maxSequenceLength: maxSequenceLength,
      batchSize: 1024))
  },
  { () in
    ("QNLI", try! QNLI(
      taskDirectoryURL: tasksDir,
      textTokenizer: textTokenizer,
      maxSequenceLength: maxSequenceLength,
      batchSize: 1024))
  },
  { () in
    ("WNLI", try! WNLI(
      taskDirectoryURL: tasksDir,
      textTokenizer: textTokenizer,
      maxSequenceLength: maxSequenceLength,
      batchSize: 1024))
  },
  { () in
    ("SNLI", try! SNLI(
      taskDirectoryURL: tasksDir,
      textTokenizer: textTokenizer,
      maxSequenceLength: maxSequenceLength,
      batchSize: 1024))
  },
  { () in
    ("MNLI", try! MNLI(
      taskDirectoryURL: tasksDir,
      textTokenizer: textTokenizer,
      maxSequenceLength: maxSequenceLength,
      batchSize: 1024))
  },
  { () in
    ("QQP", try! QQP(
      taskDirectoryURL: tasksDir,
      textTokenizer: textTokenizer,
      maxSequenceLength: maxSequenceLength,
      batchSize: 1024))
  }]
logger.info("Initializing tasks and loading their data in memory.")
var tasks = taskInitializers.concurrentMap { $0() }

var architecture = SimpleArchitecture(
  bertConfiguration: bertConfiguration,
  hiddenSize: 1024,
  contextEmbeddingSize: 16,
  reasoningHiddenSize: 1024)
try! architecture.textPerception.load(preTrainedModel: bertPreTrainedModel, from: bertDir)

var optimizer = WeightDecayedAdam(
  for: architecture,
  learningRate: ExponentiallyDecayedParameter(
    baseParameter: LinearlyWarmedUpParameter(
      baseParameter: FixedParameter(Float(2e-5)),
      warmUpStepCount: 1000,
      warmUpOffset: 0),
    decayRate: 0.999,
    decayStepCount: 1,
    startStep: 1000),
  weightDecayRate: 0.01,
  useBiasCorrection: false,
  beta1: 0.9,
  beta2: 0.999,
  epsilon: 1e-6,
  maxGradientGlobalNorm: 1.0)

logger.info("Training.")
for step in 0..<10000 {
  if step % 50 == 0 {
    // TODO: !!! Create nice table-making utilities and remove this messy temporary solution.
    logger.info("╔\([String](repeating: "═", count: 91).joined())╗")
    logger.info("║\([String](repeating: " ", count: 35).joined())\(step: step) Evaluation\([String](repeating: " ", count: 35).joined())║")
    logger.info("╠\([String](repeating: "═", count: 7).joined())╦\([String](repeating: "═", count: 32).joined())╤\([String](repeating: "═", count: 8).joined())╦\([String](repeating: "═", count: 32).joined())╤\([String](repeating: "═", count: 8).joined())╣")
    for taskIndex in tasks.indices {
      let (name, task) = tasks[taskIndex]
      if name == "MNLI" || name == "QQP" { continue }
      var results = task.evaluate(using: architecture)
        .sorted(by: { $0.key < $1.key })
        .map { "\(metricName: $0.key) │ \(metricValue: $0.value)" }
      if results.count < 2 {
        results.append("\([String](repeating: " ", count: 31).joined())│\([String](repeating: " ", count: 7).joined())")
      }
      logger.info("║ \(task: name) ║ \(results.joined(separator: " ║ ")) ║")
    }
    logger.info("╚\([String](repeating: "═", count: 7).joined())╩\([String](repeating: "═", count: 32).joined())╧\([String](repeating: "═", count: 8).joined())╩\([String](repeating: "═", count: 32).joined())╧\([String](repeating: "═", count: 8).joined())╝")
  }

  var losses = [String]()
  losses.reserveCapacity(tasks.count)
  for taskIndex in tasks.indices {
    let loss = tasks[taskIndex].1.update(architecture: &architecture, using: &optimizer)
    losses.append("\(task: tasks[taskIndex].0): \(loss: loss)")
  }

  logger.info("\(step: step) | \(losses.joined(separator: " | "))")
}
