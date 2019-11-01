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
let bertDir = modulesDir.appendingPathComponent("text").appendingPathComponent("bert")

let bert = try BERT.PreTrainedModel.robertaBase.load(from: bertDir)
//let bert = try BERT.PreTrainedModel.bertBase(cased: false, multilingual: false).load(from: bertDir)
let problemCompiler = SimpleProblemCompiler(
  problemEmbeddingSize: 32,
  conceptEmbeddingSize: 128,
  modifierEmbeddingSize: 1024,
  conceptModifierHiddenSize: 128,
  conceptModifierGeneratorHiddenSize: 512,
  problemAttentionHeadCount: 4)
var architecture = SimpleArchitecture(
  problemCompiler: problemCompiler,
  textPerception: bert,
  hiddenSize: 128,
  reasoningHiddenSize: 128)
let useCurriculum = true

let maxSequenceLength = 512 // bertConfiguration.maxSequenceLength
let taskInitializers: [() -> (String, Task)] = [
  { () in
    ("MRPC", try! MRPC(
      for: architecture,
      taskDirectoryURL: tasksDir,
      maxSequenceLength: maxSequenceLength,
      batchSize: 1024))
  },
  { () in
    ("CoLA", try! CoLA(
      for: architecture,
      taskDirectoryURL: tasksDir,
      maxSequenceLength: maxSequenceLength,
      batchSize: 1024))
  },
  { () in
    ("RTE", try! RTE(
      for: architecture,
      taskDirectoryURL: tasksDir,
      maxSequenceLength: maxSequenceLength,
      batchSize: 1024))
  },
  { () in
    ("SST", try! SST(
      for: architecture,
      taskDirectoryURL: tasksDir,
      maxSequenceLength: maxSequenceLength,
      batchSize: 1024))
  },
  { () in
    ("STS", try! STS(
      for: architecture,
      taskDirectoryURL: tasksDir,
      maxSequenceLength: maxSequenceLength,
      batchSize: 1024))
  },
  { () in
    ("QNLI", try! QNLI(
      for: architecture,
      taskDirectoryURL: tasksDir,
      maxSequenceLength: maxSequenceLength,
      batchSize: 1024))
  },
  { () in
    ("WNLI", try! WNLI(
      for: architecture,
      taskDirectoryURL: tasksDir,
      maxSequenceLength: maxSequenceLength,
      batchSize: 1024))
  },
  { () in
    ("SNLI", try! SNLI(
      for: architecture,
      taskDirectoryURL: tasksDir,
      maxSequenceLength: maxSequenceLength,
      batchSize: 1024))
  },
  { () in
    ("MNLI", try! MNLI(
      for: architecture,
      taskDirectoryURL: tasksDir,
      maxSequenceLength: maxSequenceLength,
      batchSize: 1024))
  },
  { () in
    ("QQP", try! QQP(
      for: architecture,
      taskDirectoryURL: tasksDir,
      maxSequenceLength: maxSequenceLength,
      batchSize: 1024))
  }]
logger.info("Initializing tasks and loading their data in memory.")
var tasks = taskInitializers.concurrentMap { $0() }

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
var losses = [Float](repeating: 0, count: tasks.count)
var minLosses = [Float](repeating: 0, count: tasks.count)
var lastUpdate = [Int](repeating: 0, count: tasks.count)
var updateCounts = [Int](repeating: 0, count: tasks.count)
for step in 1..<10000 {
  if useCurriculum && step == 1 {
    for taskIndex in tasks.indices {
      losses[taskIndex] = tasks[taskIndex].1.update(architecture: &architecture, using: &optimizer)
      minLosses[taskIndex] = losses[taskIndex]
      updateCounts[taskIndex] += 1
    }
    let message = zip(tasks, losses).map { "\(task: $0.0): \(loss: $1)" }.joined(separator: " | ")
    logger.info("\(step: step) | \(message)")
  } else if useCurriculum {
    let taskIndex = { () -> Int in
      var scores = zip(losses, minLosses).map { $0 - $1 }
      let minScore = scores.reduce(Float.infinity, min)
      let maxScore = scores.reduce(-Float.infinity, max)
      scores = scores.map { ($0 - minScore) / (maxScore - minScore) }
      scores = zip(scores, lastUpdate).map { $0 + Float(step - $1) * 0.001 }
      let scoresSum = scores.reduce(0, +)
      if let random = scoresSum == 0 || maxScore == minScore ? nil : Float.random(in: 0..<scoresSum) {
        var accumulator = Float(0)
        for (index, score) in scores.enumerated() {
          accumulator += score
          if random < accumulator {
            return index
          }
        }
        return scores.count - 1
      } else {
        return Int.random(in: 0..<scores.count)
      }
    }()
    let loss = tasks[taskIndex].1.update(architecture: &architecture, using: &optimizer)
    losses[taskIndex] = loss
    minLosses[taskIndex] = min(loss, minLosses[taskIndex])
    lastUpdate[taskIndex] = step
    updateCounts[taskIndex] += 1
    let message = zip(tasks, losses).enumerated().map {
      "\(task: $0 == taskIndex ? ("*" + $1.0.0) : $1.0.0): \(loss: $1.1)"
    }.joined(separator: " | ")
    logger.info("\(step: step) | \(message)")
    if step % 100 == 0 {
      let totalUpdateCount = Float(updateCounts.reduce(0, +))
      let message = zip(tasks, updateCounts).map {
        "\(task: $0.0): \(metricValue: Float($1) / totalUpdateCount)"
      }.joined(separator: " | ")
      logger.info("\(step: step) | \(message)")
    }
  } else {
    for taskIndex in tasks.indices.shuffled() {
      losses[taskIndex] = tasks[taskIndex].1.update(architecture: &architecture, using: &optimizer)
    }
    let message = zip(tasks, losses).map { "\(task: $0.0): \(loss: $1)" }.joined(separator: " | ")
    logger.info("\(step: step) | \(message)")
  }

  // Evaluation
  if step % 500 == 0 {
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
}
