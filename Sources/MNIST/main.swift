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

import Core
import Foundation
import TensorFlow

let currentDir = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
let tempDir = currentDir.appendingPathComponent("temp")
let tasksDir = tempDir.appendingPathComponent("tasks")

// let dataset = try! MNISTDataset(taskDirectoryURL: tasksDir)
let dataset = try! CIFAR10Dataset(taskDirectoryURL: tasksDir)
// let dataset = try! CIFAR100Dataset(taskDirectoryURL: tasksDir)
let batchSize = 64
let randomSeed = Int64(123456789)

// Baseline
withRandomSeedForTensorFlow(randomSeed) {
  var task = IdentityTask(
    srcModality: .image,
    tgtModality: .number,
    dataset: dataset,
    randomRotations: false,
    randomSeed: randomSeed)
  // var layer = LeNet()
  // var layer = ContextualLeNet()
  // var layer = MLP()
  var layer = ReverseContextualizedMLP(functionEmbeddingSize: 16)
  // var layer = ReverseContextualizedLeNet2(functionEmbeddingSize: 16)
  // var layer = WideResNet(kind: .wideResNet28k10)
  var optimizer = Adam(
    for: layer,
    learningRate: FixedParameter(Float(1e-3)),
    // learningRate: ExponentiallyDecayedParameter(
    //   baseParameter: LinearlyWarmedUpParameter(
    //     baseParameter: FixedParameter(Float(1e-3)),
    //     warmUpStepCount: 1000,
    //     warmUpOffset: 0),
    //   decayRate: 0.995,
    //   decayStepCount: 1,
    //   startStep: 1000),
    beta1: 0.9,
    beta2: 0.999,
    epsilon: 1e-8)

  func evaluate() -> [String: Float] {
    task.evaluate(
      layer,
      using: dataset,
      batchSize: batchSize)
  }

  print("Initial Evaluation: \(evaluate())")
  var loss: Float = 0
  for step in 0..<1000000 {
    loss += task.update(layer: &layer, using: &optimizer)
    // if step % 10 == 0 {
    //   print("Step \(step) Loss: \(loss / 10)")
    //   loss = 0
    // }
    if step % 100 == 0 {
      print("Step \(step) Evaluation: \(evaluate())")
    }
  }
}

// // NCA
// withRandomSeedForTensorFlow(randomSeed) {
//   var tasks: [Task] = [
//     IdentityTask(srcModality: .image, tgtModality: .number, dataset: dataset, randomRotations: true, randomSeed: randomSeed),
//     IdentityTask(srcModality: .number, tgtModality: .image, dataset: dataset, randomRotations: true, randomSeed: randomSeed),
//     IdentityTask(srcModality: .image, tgtModality: .image, dataset: dataset, randomRotations: true, randomSeed: randomSeed),
//     IdentityTask(srcModality: .number, tgtModality: .number, dataset: dataset, randomRotations: true, randomSeed: randomSeed),
//     RotationTask(dataset: dataset, randomSeed: randomSeed)]
//   let problemCompiler = LinearProblemCompiler(
//     problemEmbeddingSize: 4,
//     initializerStandardDeviation: 0.02)
//   var architecture = ConvolutionalArchitecture(
//     hiddenSize: 128,
//     problemCompiler: problemCompiler,
//     initializerStandardDeviation: 0.02)
//   var optimizer = Adam(
//     for: architecture,
//     learningRate: 1e-3,
//     beta1: 0.9,
//     beta2: 0.99,
//     epsilon: 1e-8,
//     decay: 0.001)

//   func evaluate() -> [String: Float] {
//     (tasks[0] as! IdentityTask).evaluate(
//       architecture,
//       using: dataset,
//       batchSize: batchSize)
//   }

//   print("Initial Evaluation: \(evaluate())")
//   var loss: Float = 0
//   for step in 0..<1000000 {
//     loss += tasks[0].update(architecture: &architecture, using: &optimizer)
//     loss += tasks[1].update(architecture: &architecture, using: &optimizer)
//     loss += tasks[2].update(architecture: &architecture, using: &optimizer)
//     loss += tasks[3].update(architecture: &architecture, using: &optimizer)
//     loss += tasks[4].update(architecture: &architecture, using: &optimizer)
//     // if step % 10 == 0 {
//     //   print("Step \(step) Loss: \(loss / 10)")
//     //   loss = 0
//     // }
//     if step % 100 == 0 {
//       print("Step \(step) Evaluation: \(evaluate())")
//     }
//   }
// }
