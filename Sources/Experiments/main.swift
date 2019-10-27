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
    appendLiteral(String(format: "%5d", value))
  }

  mutating func appendInterpolation(loss value: Float) {
    appendLiteral(String(format: "%.4f", value))
  }
}

let currentDir = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
let ncaDir = currentDir.appendingPathComponent("temp")
let modulesDir = ncaDir.appendingPathComponent("modules")
let tasksDir = ncaDir.appendingPathComponent("tasks")

let albertDir = modulesDir
  .appendingPathComponent("text")
  .appendingPathComponent("albert")
let albertPreTrainedModel = ALBERT.PreTrainedModel.base
try albertPreTrainedModel.maybeDownload(to: albertDir)

let vocabularyURL = albertDir
  .appendingPathComponent(albertPreTrainedModel.name)
  .appendingPathComponent("assets")
  .appendingPathComponent("30k-clean.model")
let vocabulary = try! Vocabulary(fromFile: vocabularyURL)
let albertConfiguration = albertPreTrainedModel.configuration
let textTokenizer = FullTextTokenizer(
  caseSensitive: false,
  vocabulary: vocabulary,
  unknownToken: "[UNK]",
  maxTokenLength: albertConfiguration.maxSequenceLength)

var mrpc = try! MRPC(
  taskDirectoryURL: tasksDir,
  textTokenizer: textTokenizer,
  maxSequenceLength: 128, // albertConfiguration.maxSequenceLength,
  batchSize: 32)
var cola = try! CoLA(
  taskDirectoryURL: tasksDir,
  textTokenizer: textTokenizer,
  maxSequenceLength: 128, // albertConfiguration.maxSequenceLength,
  batchSize: 32)
var rte = try! RTE(
  taskDirectoryURL: tasksDir,
  textTokenizer: textTokenizer,
  maxSequenceLength: 128, // albertConfiguration.maxSequenceLength,
  batchSize: 32)
var sst = try! SST(
  taskDirectoryURL: tasksDir,
  textTokenizer: textTokenizer,
  maxSequenceLength: 128, // albertConfiguration.maxSequenceLength,
  batchSize: 32)

var architecture = SimpleArchitecture(
  albertConfiguration: albertConfiguration,
  hiddenSize: 512,
  contextEmbeddingSize: 4,
  reasoningHiddenSize: 512,
  albertLearningRate: ExponentiallyDecayedParameter(
    baseParameter: LinearlyWarmedUpParameter(
      baseParameter: FixedParameter(Float(5e-5)),
      warmUpStepCount: 100,
      warmUpOffset: 0),
    decayRate: 0.99,
    decayStepCount: 1,
    startStep: 100),
  learningRate: ExponentiallyDecayedParameter(
    baseParameter: LinearlyWarmedUpParameter(
      baseParameter: FixedParameter(Float(5e-5)),
      warmUpStepCount: 100,
      warmUpOffset: 0),
    decayRate: 0.99,
    decayStepCount: 1,
    startStep: 100))
try! architecture.textPerception.load(preTrainedModel: albertPreTrainedModel, from: albertDir)

var optimizer = LAMB(
  for: architecture,
  weightDecayRate: 0.01,
  beta1: 0.9,
  beta2: 0.999,
  epsilon: 1e-6,
  maxGradientGlobalNorm: nil)

logger.info("Training is starting...")
for step in 1..<10000 {
  if step % 10 == 0 {
    let mrpcResults = mrpc.evaluate(using: architecture).summary
    let colaResults = cola.evaluate(using: architecture).summary
    let rteResults = rte.evaluate(using: architecture).summary
    let sstResults = sst.evaluate(using: architecture).summary
    let results =
      """
      ================
      Step \(step) Evaluation
      ================
      MRPC Evaluation: \(mrpcResults)
      CoLA Evaluation: \(colaResults)
      RTE Evaluation: \(rteResults)
      SST Evaluation: \(sstResults)
      ================
      """
    logger.info("\(results)")
  }
  let (mrpcLoss, mrpcGradient) = mrpc.loss(architecture: architecture)
  architecture.update(along: optimizer.update(for: architecture, along: mrpcGradient))
  let (colaLoss, colaGradient) = cola.loss(architecture: architecture)
  architecture.update(along: optimizer.update(for: architecture, along: colaGradient))
  let (rteLoss, rteGradient) = rte.loss(architecture: architecture)
  architecture.update(along: optimizer.update(for: architecture, along: rteGradient))
  let (sstLoss, sstGradient) = sst.loss(architecture: architecture)
  architecture.update(along: optimizer.update(for: architecture, along: sstGradient))
  logger.info("Step \(step: step) | MRPC Loss = \(loss: mrpcLoss) | CoLA Loss = \(loss: colaLoss) | RTE Loss = \(loss: rteLoss) | SST Loss = \(loss: sstLoss)")
}
