// Copyright 2020, Emmanouil Antonios Platanios. All Rights Reserved.
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
import TensorFlow

let currentDir = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
let tempDir = currentDir.appendingPathComponent("temp")
let tasksDir = tempDir.appendingPathComponent("tasks")

let dataset = try! Dataset(taskDirectoryURL: tasksDir)
let batchSize = 32

var generator = PhiloxRandomNumberGenerator(seed: 123456789)

print("#Train: \(dataset.partitions[.train]!.count)")
print("#Test: \(dataset.partitions[.test]!.count)")

var tasks = [
  Task(srcModality: .image, tgtModality: .number, problem: .identity, dataset: dataset)]

let problemCompiler = LinearProblemCompiler(
  problemEmbeddingSize: 4,
  initializerStandardDeviation: 0.02)
var architecture = ConvolutionalArchitecture(
  hiddenSize: 4,
  problemCompiler: problemCompiler,
  initializerStandardDeviation: 0.02)
var optimizer = Adam(
  for: architecture,
  learningRate: 1e-3,
  beta1: 0.9,
  beta2: 0.99,
  epsilon: 1e-8,
  decay: 0)

func evaluate() {
  let result = tasks[0].evaluate(architecture, using: dataset, batchSize: batchSize)
  print(result)
}

let result = tasks[0].evaluate(architecture, using: dataset, batchSize: batchSize)
print("Initial Evaluation: \(result)")
var loss: Float = 0
for step in 0..<10000 {
  loss += tasks[0].update(architecture: &architecture, using: &optimizer)
  // if step % 10 == 0 {
  //   print("Step \(step) Loss: \(loss / 10)")
  //   loss = 0
  // }
  if step % 100 == 0 {
    let result = tasks[0].evaluate(architecture, using: dataset, batchSize: batchSize)
    print("Step \(step) Evaluation: \(result)")
  }
}
