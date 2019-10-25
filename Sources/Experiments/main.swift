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

import NCA
import Foundation
import TensorFlow

let currentDir = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
let ncaDir = currentDir.appendingPathComponent("temp")
let modulesDir = ncaDir.appendingPathComponent("modules")
let bertDir = modulesDir
  .appendingPathComponent("text")
  .appendingPathComponent("bert")
let tasksDir = ncaDir.appendingPathComponent("tasks")

let bertPreTrainedModel = BERT.PreTrainedModel.base(cased: false, multilingual: false)
try bertPreTrainedModel.maybeDownload(to: bertDir)

let vocabularyURL = bertDir
  .appendingPathComponent(bertPreTrainedModel.name)
  .appendingPathComponent("vocab.txt")
let bertConfigurationURL = bertDir
  .appendingPathComponent(bertPreTrainedModel.name)
  .appendingPathComponent("bert_config.json")

let vocabulary = try! Vocabulary(fromFile: vocabularyURL)
let bertConfiguration = try! BERT.Configuration(fromFile: bertConfigurationURL)
//let bertConfiguration = BERT.Configuration(
//  vocabularySize: vocabulary.count,
//  hiddenSize: 64,
//  hiddenLayerCount: 1,
//  attentionHeadCount: 4,
//  intermediateSize: 4,
//  intermediateActivation: .gelu,
//  hiddenDropoutProbability: 0.1,
//  attentionDropoutProbability: 0.1,
//  maxSequenceLength: 50,
//  typeVocabularySize: 2,
//  initializerStandardDeviation: 0.02)
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

var architecture = SimpleArchitecture(
  bertConfiguration: bertConfiguration,
  hiddenSize: 512,
  contextEmbeddingSize: 4,
  reasoningHiddenSize: 512)
try! architecture.textPerception.load(
  preTrainedModel: .base(cased: false, multilingual: false),
  from: bertDir)

var mrpcOptimizer = WeightDecayedAdam(
  for: architecture,
  learningRate: ExponentiallyDecayedParameter(
    baseParameter: LinearlyWarmedUpParameter(
      baseParameter: FixedParameter(Float(2e-5)),
      warmUpStepCount: 14,
      warmUpOffset: 0),
    decayRate: 0.99,
    decayStepCount: 1,
    startStep: 1000),
  weightDecayRate: 0.01,
  useBiasCorrection: false,
  beta1: 0.9,
  beta2: 0.999,
  epsilon: 1e-6)
var colaOptimizer = WeightDecayedAdam(
  for: architecture,
  learningRate: ExponentiallyDecayedParameter(
    baseParameter: LinearlyWarmedUpParameter(
      baseParameter: FixedParameter(Float(2e-5)),
      warmUpStepCount: 14,
      warmUpOffset: 0),
    decayRate: 0.99,
    decayStepCount: 1,
    startStep: 1000),
  weightDecayRate: 0.01,
  useBiasCorrection: false,
  beta1: 0.9,
  beta2: 0.999,
  epsilon: 1e-6)

for step in 1..<10000 {
  print("Step \(step)")
  if step % 10 == 0 {
    let mrpcResults = mrpc.evaluate(using: architecture).summary
    let colaResults = cola.evaluate(using: architecture).summary
    let results =
      """
      ================
      Evaluation
      ================
      MRPC Evaluation: \(mrpcResults)
      CoLA Evaluation: \(colaResults)
      ================
      """
    print(results)
  }
  let mrpcLoss = mrpc.update(architecture: &architecture, using: &mrpcOptimizer)
  let colaLoss = cola.update(architecture: &architecture, using: &colaOptimizer)
  print("\tMRPC Loss = \(mrpcLoss) | CoLA Loss = \(colaLoss)")
}
