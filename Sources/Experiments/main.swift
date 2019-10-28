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

var mrpc = try! MRPC(
  taskDirectoryURL: tasksDir,
  textTokenizer: textTokenizer,
  maxSequenceLength: 128, // bertConfiguration.maxSequenceLength,
  batchSize: 32)
var cola = try! CoLA(
  taskDirectoryURL: tasksDir,
  textTokenizer: textTokenizer,
  maxSequenceLength: 128, // bertConfiguration.maxSequenceLength,
  batchSize: 32)
var rte = try! RTE(
  taskDirectoryURL: tasksDir,
  textTokenizer: textTokenizer,
  maxSequenceLength: 128, // bertConfiguration.maxSequenceLength,
  batchSize: 32)
var sst = try! SST(
  taskDirectoryURL: tasksDir,
  textTokenizer: textTokenizer,
  maxSequenceLength: 128, // bertConfiguration.maxSequenceLength,
  batchSize: 32)
var qnli = try! QNLI(
  taskDirectoryURL: tasksDir,
  textTokenizer: textTokenizer,
  maxSequenceLength: 128, // bertConfiguration.maxSequenceLength,
  batchSize: 32)
var wnli = try! WNLI(
  taskDirectoryURL: tasksDir,
  textTokenizer: textTokenizer,
  maxSequenceLength: 128, // bertConfiguration.maxSequenceLength,
  batchSize: 32)
var mnli = try! MNLI(
  taskDirectoryURL: tasksDir,
  textTokenizer: textTokenizer,
  maxSequenceLength: 128, // bertConfiguration.maxSequenceLength,
  batchSize: 32)
var qqp = try! QQP(
  taskDirectoryURL: tasksDir,
  textTokenizer: textTokenizer,
  maxSequenceLength: 128, // bertConfiguration.maxSequenceLength,
  batchSize: 32)

var architecture = SimpleArchitecture(
  bertConfiguration: bertConfiguration,
  hiddenSize: 512,
  contextEmbeddingSize: 32,
  reasoningHiddenSize: 512)
try! architecture.textPerception.load(preTrainedModel: bertPreTrainedModel, from: bertDir)

var optimizer = WeightDecayedAdam(
  for: architecture,
  learningRate: ExponentiallyDecayedParameter(
    baseParameter: LinearlyWarmedUpParameter(
      baseParameter: FixedParameter(Float(2e-5)),
      warmUpStepCount: 1000,
      warmUpOffset: 0),
    decayRate: 0.995,
    decayStepCount: 1,
    startStep: 1000),
  weightDecayRate: 0.01,
  useBiasCorrection: false,
  beta1: 0.9,
  beta2: 0.999,
  epsilon: 1e-6,
  maxGradientGlobalNorm: 1.0)

logger.info("Training is starting...")
for step in 1..<10000 {
  if step % 10 == 0 {
    let mrpcResults = mrpc.evaluate(using: architecture).summary
    let colaResults = cola.evaluate(using: architecture).summary
    let rteResults = rte.evaluate(using: architecture).summary
    let sstResults = sst.evaluate(using: architecture).summary
    let qnliResults = qnli.evaluate(using: architecture).summary
    let wnliResults = wnli.evaluate(using: architecture).summary
    let mnliResults = mnli.evaluate(using: architecture).summary
    let qqpResults = qqp.evaluate(using: architecture).summary
    let results =
      """
      ================
      Step \(step) Evaluation
      ================
      MRPC | \(mrpcResults)
      CoLA | \(colaResults)
       RTE | \(rteResults)
       SST | \(sstResults)
      QNLI | \(qnliResults)
      WNLI | \(wnliResults)
      MNLI | \(mnliResults)
       QQP | \(qqpResults)
      ================
      """
    logger.info("\(results)")
  }
  let mrpcLoss = mrpc.update(architecture: &architecture, using: &optimizer)
  let colaLoss = cola.update(architecture: &architecture, using: &optimizer)
  let rteLoss = rte.update(architecture: &architecture, using: &optimizer)
  let sstLoss = sst.update(architecture: &architecture, using: &optimizer)
  let qnliLoss = qnli.update(architecture: &architecture, using: &optimizer)
  let wnliLoss = wnli.update(architecture: &architecture, using: &optimizer)
  let mnliLoss = mnli.update(architecture: &architecture, using: &optimizer)
  let qqpLoss = qqp.update(architecture: &architecture, using: &optimizer)
  let message = "Step \(step: step) | " +
    "MRPC: \(loss: mrpcLoss) | " +
    "CoLA: \(loss: colaLoss) | " +
    "RTE: \(loss: rteLoss) | " +
    "SST: \(loss: sstLoss) | " +
    "QNLI: \(loss: qnliLoss) |" +
    "WNLI: \(loss: wnliLoss) |" +
    "MNLI: \(loss: mnliLoss) |" +
    "QQP: \(loss: qqpLoss)"
  logger.info("\(message)")
}
