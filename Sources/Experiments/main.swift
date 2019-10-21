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

let currentDir = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
let ncaDir = currentDir.appendingPathComponent("temp")
let modulesDir = ncaDir.appendingPathComponent("modules")
let bertDir = modulesDir
  .appendingPathComponent("text")
  .appendingPathComponent("bert")
  .appendingPathComponent("multi_cased_L-12_H-768_A-12")
let tasksDir = ncaDir.appendingPathComponent("tasks")

let vocabulary = try! Vocabulary(fromFile: bertDir.appendingPathComponent("vocab.txt"))
let textTokenizer = FullTextTokenizer(
  caseSensitive: false,
  vocabulary: vocabulary,
  unknownToken: "[UNK]",
  maxTokenLength: 200)

//    let squad = try! SQuAD(version: .v1_1, taskDirectoryURL: tasksDir)
let cola = try! CoLA(
  taskDirectoryURL: tasksDir,
  textTokenizer: textTokenizer,
  maxSequenceLength: 200, // TODO: !!!
  batchSize: 32)
//    let colaOptimizer = Adam(
//      for: architecture,
//      learningRate: 1e-3,
//      beta1: 0.9,
//      beta2: 0.99,
//      epsilon: 1e-8,
//      decay: 0)

let bertConfiguration = try! BERT<Float>.Configuration(
  fromFile: bertDir.appendingPathComponent("bert_config.json"))
var bert = BERT(configuration: bertConfiguration)
print(bertDir.appendingPathComponent("bert_model.ckpt").path)
bert.load(fromTensorFlowCheckpoint: bertDir.appendingPathComponent("bert_model.ckpt"))

dump(bertConfiguration)
