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
  .appendingPathComponent("multi_cased_L-12_H-768_A-12")
let tasksDir = ncaDir.appendingPathComponent("tasks")

let vocabularyURL = bertDir.appendingPathComponent("vocab.txt")
let bertConfigurationURL = bertDir.appendingPathComponent("bert_config.json")
let bertCheckpointURL = bertDir.appendingPathComponent("bert_model.ckpt")

let vocabulary = try! Vocabulary(fromFile: vocabularyURL)
let bertConfiguration = try! BERT.Configuration(fromFile: bertConfigurationURL)
let textTokenizer = FullTextTokenizer(
  caseSensitive: false,
  vocabulary: vocabulary,
  unknownToken: "[UNK]",
  maxTokenLength: bertConfiguration.maxSequenceLength)

var cola = try! CoLA(
  taskDirectoryURL: tasksDir,
  textTokenizer: textTokenizer,
  maxSequenceLength: bertConfiguration.maxSequenceLength,
  batchSize: 32)

var architecture = SimpleArchitecture(bertConfiguration: bertConfiguration)
architecture.textPerception.load(fromTensorFlowCheckpoint: bertCheckpointURL)

var colaOptimizer = Adam(
  for: architecture,
  learningRate: 1e-3,
  beta1: 0.9,
  beta2: 0.99,
  epsilon: 1e-8,
  decay: 0)

for step in 1..<10000 {
  print("Step \(step)")
  if step % 100 == 0 { print(cola.evaluate(using: architecture).summary) }
  let loss = cola.update(architecture: &architecture, using: &colaOptimizer)
  print("\tLoss = \(loss)")
}
