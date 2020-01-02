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

public typealias Number = Float
public typealias Image = Tensor<Float>

public enum Partition: Hashable {
  case train, test
}

public protocol Dataset {
  var images: [Image] { get }
  var numbers: [Number] { get }
  var partitions: [Partition: [Int]] { get }
  var numberImageIndices: [Partition: [Float: [Int]]] { get }
  var exampleCount: Int { get }
}

//===------------------------------------------------------------------------------------------===//
// MNIST
//===------------------------------------------------------------------------------------------===//

public struct MNISTDataset: Dataset {
  public let directoryURL: URL
  public let images: [Image]
  public let numbers: [Number]
  public let partitions: [Partition: [Int]]
  public let numberImageIndices: [Partition: [Float: [Int]]]
  public let randomizedTestLabels: Bool

  public var exampleCount: Int { images.count }

  public init(taskDirectoryURL: URL, randomizedTestLabels: Bool = false) throws {
    self.directoryURL = taskDirectoryURL.appendingPathComponent("MNIST")
    
    let dataURL = directoryURL.appendingPathComponent("data")
    let compressedDataLocalURLs = [
      dataURL.appendingPathComponent("train-images-idx3-ubyte.gz"),
      dataURL.appendingPathComponent("train-labels-idx1-ubyte.gz"),
      dataURL.appendingPathComponent("t10k-images-idx3-ubyte.gz"),
      dataURL.appendingPathComponent("t10k-labels-idx1-ubyte.gz")]
    
    // Download the data, if necessary.
    let compressedDataRemoteURLs = [
      URL(string: String("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"))!,
      URL(string: String("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"))!,
      URL(string: String("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"))!,
      URL(string: String("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"))!]
    try zip(compressedDataRemoteURLs, compressedDataLocalURLs).forEach { (remoteURL, localURL) in
      try maybeDownload(from: remoteURL, to: localURL)
    }

    // Extract the data, if necessary.
    try compressedDataLocalURLs.forEach { localURL in
      let extractedDirectoryURL = localURL.deletingPathExtension()
      if !FileManager.default.fileExists(atPath: extractedDirectoryURL.path) {
        try extract(gZippedFileAt: localURL, to: extractedDirectoryURL)
      }
    }

    // Load the data files into arrays of examples.
    let extractedDataLocalURLs = compressedDataLocalURLs.map { $0.deletingPathExtension() }
    let trnImages = [UInt8](try Data(contentsOf: extractedDataLocalURLs[0])).dropFirst(16)
    let trnLabels = [UInt8](try Data(contentsOf: extractedDataLocalURLs[1])).dropFirst(8)
    let tstImages = [UInt8](try Data(contentsOf: extractedDataLocalURLs[2])).dropFirst(16)
    var tstLabels = [UInt8](try Data(contentsOf: extractedDataLocalURLs[3])).dropFirst(8)

    if randomizedTestLabels {
      for i in tstLabels.indices {
        tstLabels[i] = .random(in: 0...9)
      }
    }

    // Initialize this dataset instance.
    let exampleCount = trnLabels.count + tstLabels.count
    self.images = { () -> [Tensor<Float>] in
      var images = Tensor<Float>(Tensor<UInt8>(
        shape: [trnLabels.count + tstLabels.count, 28, 28, 1],
        scalars: trnImages + tstImages))
      images = images.tiled(multiples: [1, 1, 1, 3])
      images /= 255
      return images.unstacked(alongAxis: 0)
    }()
    self.numbers = [UInt8](trnLabels + tstLabels).map(Float.init)
    self.partitions = [
      .train: [Int](0..<trnLabels.count),
      .test: [Int](trnLabels.count..<exampleCount)]
    self.numberImageIndices = self.partitions.mapValues { [numbers] indices in
      [Float: [Int]](grouping: indices, by: { numbers[$0] })
    }
    self.randomizedTestLabels = randomizedTestLabels
  }
}

//===------------------------------------------------------------------------------------------===//
// CIFAR
//===------------------------------------------------------------------------------------------===//

public enum CIFARLabel: UInt8 {
  case airplane = 0, automobile, bird, cat, deer, dog, frog, horse, ship, truck
}

public struct CIFAR10Dataset: Dataset {
  public let directoryURL: URL
  public let images: [Image]
  public let numbers: [Number]
  public let partitions: [Partition: [Int]]
  public let numberImageIndices: [Partition: [Float: [Int]]]
  public let randomizedTestLabels: Bool

  public var exampleCount: Int { images.count }

  public init(taskDirectoryURL: URL, randomizedTestLabels: Bool = false) throws {
    self.directoryURL = taskDirectoryURL.appendingPathComponent("CIFAR")
    
    let dataURL = directoryURL.appendingPathComponent("data")
    let compressedDataLocalURL = dataURL.appendingPathComponent("cifar-10-binary.tar.gz")
    
    // Download the data, if necessary.
    let compressedDataRemoteURL = URL(
      string: String("https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"))!
    try maybeDownload(from: compressedDataRemoteURL, to: compressedDataLocalURL)

    // Extract the data, if necessary.
    let extractedDirectoryURL = dataURL.appendingPathComponent("cifar-10-binary")
    if !FileManager.default.fileExists(atPath: extractedDirectoryURL.path) {
      try extract(tarGZippedFileAt: compressedDataLocalURL, to: extractedDirectoryURL)
    }

    // Load the data file into arrays.
    let files = [
      "data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin",
      "data_batch_4.bin", "data_batch_5.bin", "test_batch.bin"]
    let filesURL = extractedDirectoryURL.appendingPathComponent("cifar-10-batches-bin")
    var imageBytes = [UInt8]()
    var labels = [Int64]()
    let imageByteCount = 3073
    for file in files {
      let fileContents = try! Data(contentsOf: filesURL.appendingPathComponent(file))
      let imageCount = fileContents.count / imageByteCount
      for imageIndex in 0..<imageCount {
        let baseAddress = imageIndex * imageByteCount
        imageBytes.append(contentsOf: fileContents[(baseAddress + 1)..<(baseAddress + 3073)])
        labels.append(Int64(fileContents[baseAddress]))
      }
    }

    if randomizedTestLabels {
      for i in 50000..<60000 {
        labels[i] = .random(in: 0...9)
      }
    }

    // Initialize this dataset instance.
    // let colorMean = Tensor<Float>([0.485, 0.456, 0.406])
    // let colorStd = Tensor<Float>([0.229, 0.224, 0.225])
    self.images = { () -> [Tensor<Float>] in
      var images = Tensor<Float>(Tensor<UInt8>(shape: [60000, 3, 32, 32], scalars: imageBytes))
      images = images[0..., 0..., 2..<30, 2..<30]           // Crop to 28x28.
      images = images.transposed(permutation: [0, 2, 3, 1]) // Transpose to NHWC format.
      images /= 255
      // images = (images - colorMean) / colorStd
      return images.unstacked(alongAxis: 0)
    }()
    self.numbers = labels.map(Float.init)
    self.partitions = [.train: [Int](0..<50000), .test: [Int](50000..<60000)]
    self.numberImageIndices = self.partitions.mapValues { [numbers] indices in
      [Float: [Int]](grouping: indices, by: { numbers[$0] })
    }
    self.randomizedTestLabels = randomizedTestLabels
  }
}

public struct CIFAR100Dataset: Dataset {
  public let directoryURL: URL
  public let images: [Image]
  public let numbers: [Number]
  public let partitions: [Partition: [Int]]
  public let numberImageIndices: [Partition: [Float: [Int]]]
  public let randomizedTestLabels: Bool

  public var exampleCount: Int { images.count }

  public init(taskDirectoryURL: URL, randomizedTestLabels: Bool = false) throws {
    self.directoryURL = taskDirectoryURL.appendingPathComponent("CIFAR")
    
    let dataURL = directoryURL.appendingPathComponent("data")
    let compressedDataLocalURL = dataURL.appendingPathComponent("cifar-100-binary.tar.gz")
    
    // Download the data, if necessary.
    let compressedDataRemoteURL = URL(
      string: String("https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz"))!
    try maybeDownload(from: compressedDataRemoteURL, to: compressedDataLocalURL)

    // Extract the data, if necessary.
    let extractedDirectoryURL = dataURL.appendingPathComponent("cifar-100-binary")
    if !FileManager.default.fileExists(atPath: extractedDirectoryURL.path) {
      try extract(tarGZippedFileAt: compressedDataLocalURL, to: extractedDirectoryURL)
    }

    // Load the data file into arrays.
    let files = ["train.bin", "test.bin"]
    let filesURL = extractedDirectoryURL.appendingPathComponent("cifar-100-binary")
    var imageBytes = [UInt8]()
    var labels = [Int64]()
    let imageByteCount = 3074
    for file in files {
      let fileContents = try! Data(contentsOf: filesURL.appendingPathComponent(file))
      let imageCount = fileContents.count / imageByteCount
      for imageIndex in 0..<imageCount {
        let baseAddress = imageIndex * imageByteCount
        imageBytes.append(contentsOf: fileContents[(baseAddress + 2)..<(baseAddress + 3074)])
        labels.append(Int64(fileContents[baseAddress + 1]))
      }
    }

    if randomizedTestLabels {
      for i in 50000..<60000 {
        labels[i] = .random(in: 0...100)
      }
    }

    // Initialize this dataset instance.
    let colorMean = Tensor<Float>([0.485, 0.456, 0.406])
    let colorStd = Tensor<Float>([0.229, 0.224, 0.225])
    self.images = { () -> [Tensor<Float>] in
      var images = Tensor<Float>(Tensor<UInt8>(shape: [60000, 3, 32, 32], scalars: imageBytes))
      images = images[0..., 0..., 2..<30, 2..<30]           // Crop to 28x28.
      images = images.transposed(permutation: [0, 2, 3, 1]) // Transpose to NHWC format.
      images /= 255
      images = (images - colorMean) / colorStd
      return images.unstacked(alongAxis: 0)
    }()
    self.numbers = labels.map(Float.init)
    self.partitions = [.train: [Int](0..<50000), .test: [Int](50000..<60000)]
    self.numberImageIndices = self.partitions.mapValues { [numbers] indices in
      [Float: [Int]](grouping: indices, by: { numbers[$0] })
    }
    self.randomizedTestLabels = randomizedTestLabels
  }
}
