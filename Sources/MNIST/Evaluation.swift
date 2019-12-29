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

import TensorFlow

extension Task {
  public func evaluate<A: Architecture>(
    _ architecture: A,
    using dataset: Dataset,
    batchSize: Int
  ) -> [String: Float] {
    // TODO: Support the evaluation of image generation.
    if tgtModality != .number { return [:] }

    func exampleMap(_ index: Int) -> Example {
      let input = { () -> Tensor<UInt8> in
        switch (srcModality) {
        case .image: return dataset.images[index]
        case .number: return Tensor<UInt8>(dataset.numbers[index])
        }
      }()
      let output = Tensor<UInt8>(target(for: dataset.numbers[index], problem: problem))
      return Example(input: input, output: output)
    }

    var tstExamples = dataset.partitions[.test]!
      .makeIterator()
      .map(exampleMap)
      .batched(batchSize: batchSize)
      .prefetched(count: 2)
    var correctCount = 0
    var totalCount = 0
    while let batch = tstExamples.next() {
      let predictions = { () -> Tensor<UInt8> in
        switch (srcModality) {
        case .image: return Tensor<UInt8>(architecture
          .generateNumber(forImage: batch.input, problem: problem)
          .argmax(squeezingAxis: -1))
        case .number: return Tensor<UInt8>(architecture
          .generateNumber(forNumber: batch.input, problem: problem)
          .argmax(squeezingAxis: -1))
        }
      }()
      correctCount += Int(Tensor<Int32>(predictions .== batch.output).sum().scalarized())
      totalCount += predictions.shape[0]
    }
    return ["acccuracy": Float(correctCount) / Float(totalCount)]
  }
}
