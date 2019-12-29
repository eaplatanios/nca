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
import Progress
import TensorFlow

public typealias Number = Float
public typealias Image = Tensor<Float>

public enum Partition: Hashable {
  case train, test
}

public struct Example: KeyPathIterable {
  public var input: Tensor<Float>
  public var output: Tensor<Float>

  public init(input: Tensor<Float>, output: Tensor<Float>) {
    self.input = input
    self.output = output
  }
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

//===------------------------------------------------------------------------------------------===//
// Iterators
//===------------------------------------------------------------------------------------------===//

extension IteratorProtocol {
  /// Returns an iterator that maps elements of this iterator using the provided function.
  ///
  /// - Parameters:
  ///   - mapFn: Function used to map the iterator elements.
  public func map<MappedElement>(
    _ mapFn: @escaping (Element) -> MappedElement
  ) -> MapIterator<Self, MappedElement> {
    MapIterator(self, mapFn: mapFn)
  }

  /// Returns an iterator that repeats this iterator indefinitely.
  public func repeated() -> RepeatIterator<Self> {
    RepeatIterator(self)
  }

  /// Returns an iterator that shuffles this iterator using a temporary buffer.
  ///
  /// - Parameters:
  ///   - bufferSize: Size of the shuffle buffer.
  public func shuffled(bufferSize: Int) -> ShuffleIterator<Self> {
    ShuffleIterator(self, bufferSize: bufferSize)
  }

  // TODO: [DOC] Add documentation string.
  public func grouped(
    keyFn: @escaping (Element) -> Int,
    sizeFn: @escaping (Int) -> Int,
    reduceFn: @escaping ([Element]) -> Element
  ) -> GroupedIterator<Self> {
    GroupedIterator(self, keyFn: keyFn, sizeFn: sizeFn, reduceFn: reduceFn)
  }

  // TODO: [DOC] Add documentation string.
  public func prefetched(count: Int) -> PrefetchIterator<Self> {
    PrefetchIterator(self, prefetchCount: count)
  }
}

extension IteratorProtocol where Element: KeyPathIterable {
  /// Returns an iterator that batches elements of this iterator.
  ///
  /// - Parameters:
  ///   - batchSize: Batch size.
  public func batched(batchSize: Int) -> BatchIterator<Self> {
    BatchIterator(self, batchSize: batchSize)
  }
}

/// Iterator that maps elements of another iterator using the provided function.
public struct MapIterator<Base: IteratorProtocol, MappedElement>: IteratorProtocol {
  private var iterator: Base
  private let mapFn: (Base.Element) -> MappedElement

  public init(_ iterator: Base, mapFn: @escaping (Base.Element) -> MappedElement) {
    self.iterator = iterator
    self.mapFn = mapFn
  }

  public mutating func next() -> MappedElement? {
    if let element = iterator.next() { return mapFn(element) }
    return nil
  }
}

/// Iterator that repeats another iterator indefinitely.
public struct RepeatIterator<Base: IteratorProtocol>: IteratorProtocol {
  private let originalIterator: Base
  private var currentIterator: Base

  public init(_ iterator: Base) {
    self.originalIterator = iterator
    self.currentIterator = iterator
  }

  public mutating func next() -> Base.Element? {
    if let element = currentIterator.next() {
      return element
    }
    currentIterator = originalIterator
    return currentIterator.next()
  }
}

/// Iterator that shuffles another iterator using a temporary buffer.
public struct ShuffleIterator<Base: IteratorProtocol>: IteratorProtocol {
  private let bufferSize: Int
  private var iterator: Base
  private var buffer: [Base.Element]
  private var bufferIndex: Int

  public init(_ iterator: Base, bufferSize: Int) {
    self.bufferSize = bufferSize
    self.iterator = iterator
    self.buffer = []
    self.bufferIndex = 0
  }

  public mutating func next() -> Base.Element? {
    if buffer.isEmpty || (bufferIndex >= bufferSize && bufferSize != -1) { fillBuffer() }
    if buffer.isEmpty { return nil }
    bufferIndex += 1
    return buffer[bufferIndex - 1]
  }

  private mutating func fillBuffer() {
    buffer = []
    bufferIndex = 0
    while let element = iterator.next(), bufferIndex < bufferSize || bufferSize == -1 {
      buffer.append(element)
      bufferIndex += 1
    }
    bufferIndex = 0
  }
}

/// Iterator that batches elements from another iterator.
public struct BatchIterator<Base: IteratorProtocol>: IteratorProtocol
where Base.Element: KeyPathIterable {
  private let batchSize: Int
  private var iterator: Base
  private var buffer: [Base.Element]

  public init(_ iterator: Base, batchSize: Int) {
    self.batchSize = batchSize
    self.iterator = iterator
    self.buffer = []
    self.buffer.reserveCapacity(batchSize)
  }

  public mutating func next() -> Base.Element? {
    while buffer.count < batchSize {
      if let element = iterator.next() {
        buffer.append(element)
      } else {
        break
      }
    }
    if buffer.isEmpty { return nil }
    let batch = Base.Element.batch(buffer)
    buffer = []
    buffer.reserveCapacity(batchSize)
    return batch
  }
}

/// Iterator that groupes elements from another iterator.
public struct GroupedIterator<Base: IteratorProtocol>: IteratorProtocol {
  private let keyFn: (Base.Element) -> Int
  private let sizeFn: (Int) -> Int
  private let reduceFn: ([Base.Element]) -> Base.Element
  private var iterator: Base
  private var groups: [Int: [Base.Element]]
  private var currentGroup: Dictionary<Int, [Base.Element]>.Index? = nil

  public init(
    _ iterator: Base,
    keyFn: @escaping (Base.Element) -> Int,
    sizeFn: @escaping (Int) -> Int,
    reduceFn: @escaping ([Base.Element]) -> Base.Element
  ) {
    self.keyFn = keyFn
    self.sizeFn = sizeFn
    self.reduceFn = reduceFn
    self.iterator = iterator
    self.groups = [Int: [Base.Element]]()
  }

  public mutating func next() -> Base.Element? {
    var elements: [Base.Element]? = nil
    while elements == nil {
      if let element = iterator.next() {
        let key = keyFn(element)
        if !groups.keys.contains(key) {
          groups[key] = [element]
        } else {
          groups[key]!.append(element)
        }
        if groups[key]!.count >= sizeFn(key) {
          elements = groups.removeValue(forKey: key)!
        }
      } else {
        break
      }
    }
    guard let elementsToReduce = elements else {
      if currentGroup == nil { currentGroup = groups.values.startIndex }
      if currentGroup! >= groups.values.endIndex { return nil }
      while groups.values[currentGroup!].isEmpty {
        currentGroup = groups.values.index(after: currentGroup!)
      }
      let elementsToReduce = groups.values[currentGroup!]
      currentGroup = groups.values.index(after: currentGroup!)
      return reduceFn(elementsToReduce)
    }
    return reduceFn(elementsToReduce)
  }
}

/// Iterator that prefetches elements from another iterator asynchronously.
public struct PrefetchIterator<Base: IteratorProtocol>: IteratorProtocol {
  private let iterator: Base
  private let prefetchCount: Int

  private var queue: BlockingQueue<Base.Element>

  public init(_ iterator: Base, prefetchCount: Int) {
    self.iterator = iterator
    self.prefetchCount = prefetchCount
    self.queue = BlockingQueue<Base.Element>(count: prefetchCount, iterator: iterator)
  }

  public mutating func next() -> Base.Element? {
    queue.read()
  }

  // TODO: !!! This is needed because `BlockingQueue` is a class. Figure out a better solution.
  public func copy() -> PrefetchIterator {
    PrefetchIterator(iterator, prefetchCount: prefetchCount)
  }
}

extension PrefetchIterator {
  internal class BlockingQueue<Element> {
    private let prefetchingDispatchQueue: DispatchQueue = DispatchQueue(label: "PrefetchIterator")
    private let writeSemaphore: DispatchSemaphore
    private let readSemaphore: DispatchSemaphore
    private let deletedSemaphore: DispatchSemaphore
    private let dispatchQueue: DispatchQueue
    private var array: [Element?]
    private var readIndex: Int
    private var writeIndex: Int
    private var depleted: Bool
    private var deleted: Bool

    internal init<Base: IteratorProtocol>(
      count: Int,
      iterator: Base
    ) where Base.Element == Element {
      self.writeSemaphore = DispatchSemaphore(value: count)
      self.readSemaphore = DispatchSemaphore(value: 0)
      self.deletedSemaphore = DispatchSemaphore(value: 0)
      self.dispatchQueue = DispatchQueue(label: "BlockingQueue")
      self.array = [Element?](repeating: nil, count: count)
      self.readIndex = 0
      self.writeIndex = 0
      self.depleted = false
      self.deleted = false
      var iterator = iterator
      prefetchingDispatchQueue.async { [unowned self] () in
        while !self.deleted {
          if let element = iterator.next() {
            self.write(element)
          } else {
            self.depleted = true
            self.readSemaphore.signal()
            self.deletedSemaphore.signal()
            break
          }
        }
        self.readSemaphore.signal()
        self.deletedSemaphore.signal()
      }
    }

    deinit {
      self.deleted = true

      // Signal the write semaphore to make sure it's not in use anymore. It's final value must be
      // greater or equal to its initial value.
      for _ in 0...array.count { writeSemaphore.signal() }

      // Wait for the delete semaphore to make sure the prefetching thread is done.
      deletedSemaphore.wait()
    }

    private func write(_ element: Element) {
      writeSemaphore.wait()
      dispatchQueue.sync {
        array[writeIndex % array.count] = element
        writeIndex += 1
      }
      readSemaphore.signal()
    }

    internal func read() -> Element? {
      if self.depleted { return nil }
      readSemaphore.wait()
      let element = dispatchQueue.sync { () -> Element? in
        let element = array[readIndex % array.count]
        array[readIndex % array.count] = nil
        readIndex += 1
        return element
      }
      writeSemaphore.signal()
      return element
    }
  }
}

//===------------------------------------------------------------------------------------------===//
// Utilities
//===------------------------------------------------------------------------------------------===//

#if os(Linux)
import FoundationNetworking
#endif

/// Downloads the file at `url` to `path`, if `path` does not exist.
///
/// - Parameters:
///   - from: URL to download data from.
///   - to: Destination file path.
///
/// - Returns: Boolean value indicating whether a download was
///     performed (as opposed to not needed).
internal func maybeDownload(from url: URL, to destination: URL) throws {
  if !FileManager.default.fileExists(atPath: destination.path) {
    // Create any potentially missing directories.
    try FileManager.default.createDirectory(
      atPath: destination.deletingLastPathComponent().path,
      withIntermediateDirectories: true)

    // Create the URL session that will be used to download the dataset.
    let semaphore = DispatchSemaphore(value: 0)
    let delegate = DataDownloadDelegate(destinationFileUrl: destination, semaphore: semaphore)
    let session = URLSession(configuration: .ephemeral, delegate: delegate, delegateQueue: nil)

    // Download the data to a temporary file and then copy that file to
    // the destination path.
    logger.info("Downloading \(url).")
    let task = session.downloadTask(with: url)
    task.resume()

    // Wait for the download to finish.
    semaphore.wait()
  }
}

internal class DataDownloadDelegate: NSObject, URLSessionDownloadDelegate {
  let destinationFileUrl: URL
  let semaphore: DispatchSemaphore
  let numBytesFrequency: Int64

  internal var logCount: Int64 = 0
  internal var progressBar: ProgressBar? = nil

  init(
    destinationFileUrl: URL,
    semaphore: DispatchSemaphore,
    numBytesFrequency: Int64 = 1024 * 1024
  ) {
    self.destinationFileUrl = destinationFileUrl
    self.semaphore = semaphore
    self.numBytesFrequency = numBytesFrequency
  }

  internal func urlSession(
    _ session: URLSession,
    downloadTask: URLSessionDownloadTask,
    didWriteData bytesWritten: Int64,
    totalBytesWritten: Int64,
    totalBytesExpectedToWrite: Int64
  ) -> Void {
    if progressBar == nil {
      progressBar = ProgressBar(
        count: Int(totalBytesExpectedToWrite) / (1024 * 1024),
        configuration: [
          ProgressString(string: "Download Progress (MBs):"),
          ProgressIndex(),
          ProgressBarLine(),
          ProgressTimeEstimates()])
    }
    progressBar!.setValue(Int(totalBytesWritten) / (1024 * 1024))
  }

  internal func urlSession(
    _ session: URLSession,
    downloadTask: URLSessionDownloadTask,
    didFinishDownloadingTo location: URL
  ) -> Void {
    do {
      try FileManager.default.moveItem(at: location, to: destinationFileUrl)
    } catch (let writeError) {
      logger.error("Error writing file \(location.path) : \(writeError)")
    }
    logger.info("Downloaded successfully to \(location.path).")
    semaphore.signal()
  }
}

internal func extract(zipFileAt source: URL, to destination: URL) throws {
  logger.info("Extracting file at '\(source.path)'.")
  let process = Process()
  process.environment = ProcessInfo.processInfo.environment
  process.executableURL = URL(fileURLWithPath: "/bin/bash")
  process.arguments = ["-c", "unzip -d \(destination.path) \(source.path)"]
  try process.run()
  process.waitUntilExit()
}

internal func extract(gZippedFileAt source: URL, to destination: URL) throws {
  logger.info("Extracting file at '\(source.path)'.")
  let process = Process()
  process.environment = ProcessInfo.processInfo.environment
  process.executableURL = URL(fileURLWithPath: "/bin/bash")
  process.arguments = ["-c", "gunzip -c \(source.path) > \(destination.path)"]
  try process.run()
  process.waitUntilExit()
}

internal func extract(tarGZippedFileAt source: URL, to destination: URL) throws {
  logger.info("Extracting file at '\(source.path)'.")
  try FileManager.default.createDirectory(
    at: destination,
    withIntermediateDirectories: false)
  let process = Process()
  process.environment = ProcessInfo.processInfo.environment
  process.executableURL = URL(fileURLWithPath: "/bin/bash")
  process.arguments = ["-c", "tar -C \(destination.path) -xzf \(source.path)"]
  try process.run()
  process.waitUntilExit()
}
